"""Reconstruct a local raw ATD cache for Lyna's smooth50 script.

From the repository root, run:
    python -B -m experiments.reconstruct_tactical_atd
    python -B experiments/lyna_nearest_smooth50.py

The reconstruction uses the licensed tactical XML and TXT in data/tactical
and checks the graded frames against the existing cache/ball_trajectory.npz.
It writes only a new cache/tactical_atd.npz. Both arrays store float64
triples in raw units: normalized x/y and km/h. No clipping, interpolation,
or half-wise coordinate transformation is applied. This is a direct export
of the local inputs; Lyna's original cache provenance is unknown.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from games import GAME, GAMES, half_video_blocks


TRACKING = Path("data/tactical/BAR-ATH_tactical_atd")
REFERENCE = Path("cache/ball_trajectory.npz")
OUTPUT = Path("cache/tactical_atd.npz")
N_FRAMES = GAMES[GAME]["n_video_frames"]
N_SLOTS = 46
CHANNELS = (("position_x", "normalized"), ("position_y", "normalized"), ("speed_kph", "kph"))


def validate_metadata(path):
    root = ET.parse(path).getroot()
    spec = root.find(".//DataFormatSpecification")
    if spec is None or spec.attrib != {
        "startFrame": "1", "endFrame": str(N_FRAMES), "separator": ":"
    }:
        raise ValueError("Unexpected XML frame range or frame separator")

    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    players = list(root.iter("Player"))
    if [name for _, name in teams] != ["Barcelona", "Atletic Club"]:
        raise ValueError("Unexpected tactical home/away team order")
    if [p.get("teamId") for p in players] != [teams[0][0]] * 24 + [teams[1][0]] * 22:
        raise ValueError("XML player slots do not match the 24 home / 22 away split")

    channels = {
        c.get("id"): (c.findtext("Name"), c.findtext("Unit"))
        for c in root.iter("Channel")
    }
    player_channels = {
        p.get("id"): (p.get("playerId"), channels[p.get("channelId")])
        for p in root.iter("PlayerChannel")
    }
    player_register, ball_register = spec.findall("SplitRegister")
    if player_register.get("separator") != ";" or ball_register.get("separator") != ",":
        raise ValueError("Unexpected XML player or ball separator")
    expected_slots = [
        [(player.get("id"), channel) for channel in CHANNELS] for player in players
    ]
    declared_slots = [
        [player_channels[ref.get("playerChannelId")] for ref in group]
        for group in player_register.findall("SplitRegister")
    ]
    if declared_slots != expected_slots or any(
        group.get("separator") != "," for group in player_register
    ):
        raise ValueError("XML raw player registers do not match slot and channel order")
    declared_ball = tuple(channels[ref.get("channelId")] for ref in ball_register)
    if declared_ball != CHANNELS:
        raise ValueError("Unexpected XML ball channel order or units")


def read_tracking(path):
    players = np.empty((N_FRAMES, N_SLOTS, 3), dtype=np.float64)
    ball = np.empty((N_FRAMES, 3), dtype=np.float64)
    n_rows = 0
    with path.open(encoding="utf-8") as handle:
        for n_rows, line in enumerate(handle, 1):
            try:
                frame_id, player_text, ball_text = line.strip().split(":")
                index = int(frame_id) - 1
                if n_rows > N_FRAMES or index != n_rows - 1:
                    raise ValueError("unexpected frame ID")
                groups = player_text.split(";")
                ball_fields = ball_text.split(",")
                if (
                    len(groups) != N_SLOTS
                    or any(group.count(",") != 2 for group in groups)
                    or len(ball_fields) != 3
                ):
                    raise ValueError("expected 46 player triples and one ball triple")
                fields = player_text.replace(";", ",").split(",")
                players[index] = np.asarray(fields, dtype=np.float64).reshape(N_SLOTS, 3)
                ball[index] = np.asarray(ball_fields, dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f"Invalid tactical record at line {n_rows}") from exc
    if n_rows != N_FRAMES:
        raise ValueError(f"Expected {N_FRAMES} tactical rows, found {n_rows}")
    if np.isinf(players).any() or np.isinf(ball).any():
        raise ValueError("Unexpected infinity in the raw tracking data")
    return players, ball


def check_ball_reference(ball, path):
    blocks = half_video_blocks(GAME)
    expected_frames = np.concatenate([np.arange(a, b + 1) for _, a, b in blocks])
    with np.load(path) as ref:
        if not np.array_equal(ref["frame"], expected_frames):
            raise ValueError("Ball trajectory cache does not cover both graded windows in order")
        ref_position, ref_visible = ref["position"], ref["visible"]
    if ref_position.shape != (len(expected_frames), 2) or ref_visible.shape != expected_frames.shape:
        raise ValueError("Unexpected shape in the ball trajectory cache")

    xy = ball[expected_frames, :2] * GAMES[GAME]["pitch"]
    same_xy = (xy == ref_position) | (np.isnan(xy) & np.isnan(ref_position))
    same_visible = (~np.isnan(xy).any(axis=1)) == ref_visible
    offset = 0
    for half, first, last in blocks:
        length = last - first + 1
        segment = slice(offset, offset + length)
        print(
            f"Half {half}: {length:,} ball frames checked, "
            f"{np.count_nonzero(~same_xy[segment].all(axis=1)):,} position mismatches, "
            f"{np.count_nonzero(~same_visible[segment]):,} visibility mismatches"
        )
        offset += length
    if not same_xy.all() or not same_visible.all():
        raise ValueError("Ball cross-check failed; cache was not written")


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists: {OUTPUT}")
    validate_metadata(TRACKING.with_suffix(".xml"))
    print("Verified XML channel order and slot split (24 home, 22 away).", flush=True)
    players, ball = read_tracking(TRACKING.with_suffix(".txt"))
    print(f"Validated all frame IDs 1..{N_FRAMES} and converted to zero-based rows.")
    for name, array in (("players", players), ("ball", ball)):
        missing = np.isnan(array)
        visible = ~missing[..., :2].any(axis=-1)
        outside = visible & ((array[..., :2] < 0) | (array[..., :2] > 1)).any(axis=-1)
        print(
            f"{name}: {array.shape}, {array.dtype}, NaNs by x/y/speed "
            f"{tuple(int(missing[..., i].sum()) for i in range(3))}, "
            f"{np.count_nonzero(outside):,} visible out-of-pitch positions"
        )
    check_ball_reference(ball, REFERENCE)

    handle = OUTPUT.open("xb")
    try:
        with handle:
            np.savez_compressed(handle, players=players, ball=ball)
    except BaseException:
        OUTPUT.unlink()
        raise
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
