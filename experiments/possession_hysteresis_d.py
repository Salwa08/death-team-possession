import argparse
import csv
import json
from pathlib import Path
 
import numpy as np
 
from evaluation import possession_metrics as pm
from experiments.dead_mask import build_dead_mask
from games import GAME, asset_paths, half_video_blocks, source_frames
 
 
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FROZEN = RESULTS / "possession_v1.csv"
BASELINE = ROOT / "evaluation/results/baseline_smooth50.csv"
 
# Fixed engineering assumptions at 25 fps. These are uncalibrated proxies.
CONTROL_DISTANCE_M = 3.0
DISTANCE_ADVANTAGE_M = 1.0
RELATIVE_SPEED_MPS = 3.0
ACQUIRE_FRAMES = 5
SWITCH_FRAMES = 10
HOLD_FRAMES = 25
NO_TEAM = -1
 
 
def team_votes(distance, relative_speed, visible, outside):
    """Return H/A support or NO_TEAM from metre and m/s tracking arrays."""
    votes = np.full(len(visible), NO_TEAM, dtype=np.int8)
    observed = visible & ~outside & np.isfinite(distance).all(axis=1)
    for team, opponent in ((pm.H, pm.A), (pm.A, pm.H)):
        advantage = distance[:, opponent] - distance[:, team]
        support = (
            observed
            & (distance[:, team] <= CONTROL_DISTANCE_M)
            & (advantage >= DISTANCE_ADVANTAGE_M)
            & np.isfinite(relative_speed[:, team])
            & (relative_speed[:, team] <= RELATIVE_SPEED_MPS)
        )
        votes[support] = team
    return votes
 
 
def nearest_votes(distance, visible, outside):
    """Weak restart guess: the team nearest the ball, or NO_TEAM if unobserved."""
    votes = np.full(len(visible), NO_TEAM, dtype=np.int8)
    observed = visible & ~outside & np.isfinite(distance).all(axis=1)
    nearest = np.where(distance[:, pm.H] < distance[:, pm.A], pm.H, pm.A)
    votes[observed] = nearest[observed]
    return votes
 
 
def decode(frames, votes, outside, dead, restart_votes=None):
    """Confirm a team with consecutive votes and expire stale control.
 
    A discontinuity, an outside observation, a dead-ball frame or expiry clears
    team memory. After a stop, or at the start of a half, the first frame with
    a nearest-team guess assigns the owner at once, because an abstention there
    would let the scorer carry the preceding D label forward.
    """
    halt = outside | dead
    labels = np.full(len(frames), pm.X, dtype=np.int8)
    owner = pending = NO_TEAM
    streak = 0
    last_support = 0
    after_stop = True
    for i, frame in enumerate(frames):
        if (
            i == 0
            or frame != frames[i - 1] + 1
            or halt[i]
            or (owner != NO_TEAM and frame - last_support > HOLD_FRAMES)
        ):
            if i == 0 or frame != frames[i - 1] + 1 or halt[i]:
                after_stop = True
            owner = pending = NO_TEAM
            streak = 0
 
        if halt[i]:
            labels[i] = pm.D if dead[i] else pm.X
            continue
 
        if (
            owner == NO_TEAM
            and after_stop
            and restart_votes is not None
            and restart_votes[i] != NO_TEAM
        ):
            owner = restart_votes[i]
            last_support = frame
            after_stop = False
 
        vote = votes[i]
        if vote == NO_TEAM or vote == owner:
            pending = NO_TEAM
            streak = 0
            if vote != NO_TEAM:
                last_support = frame
        else:
            streak = streak + 1 if vote == pending else 1
            pending = vote
            required = ACQUIRE_FRAMES if owner == NO_TEAM else SWITCH_FRAMES
            if streak >= required:
                owner = vote
                last_support = frame
                pending = NO_TEAM
                streak = 0
 
        if owner != NO_TEAM:
            labels[i] = owner
    return labels
 
 
def load_tracking(frames):
    """Read the aligned caches produced by the frozen feature extractors."""
    with np.load(ROOT / "cache/control_evidence.npz", allow_pickle=False) as evidence:
        if not np.array_equal(evidence["frame"], frames):
            raise ValueError("Control evidence does not match the published windows")
        distance = np.column_stack([evidence["h_distance"], evidence["a_distance"]])
        relative = np.column_stack([evidence["h_relative_speed"], evidence["a_relative_speed"]])
    with np.load(ROOT / "cache/game_state_evidence.npz", allow_pickle=False) as state:
        if not np.array_equal(state["frame"], frames):
            raise ValueError("Visibility evidence does not match the published windows")
        visible = state["ball_visible"]
        outside = state["ball_outside_pitch"]
    if distance.shape != (len(frames), 2) or relative.shape != distance.shape:
        raise ValueError("Invalid control evidence shape")
    if any(mask.shape != frames.shape or mask.dtype.kind != "b" for mask in (visible, outside)):
        raise ValueError("Invalid visibility or outside mask")
    if np.any(outside & ~visible):
        raise ValueError("Outside observations require a visible ball")
    return distance, relative, visible, outside
 
 
def read_codes(path):
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, source_frames())
    if reason:
        raise ValueError(f"{path}: {reason}")
    return pm.label_codes(table)
 
 
def describe(codes, frames, runs, visible):
    """Structural diagnostics on the published windows, using the scorer's fill."""
    raw = codes[frames]
    filled = pm.fill_abstentions(codes, runs)
    durations = [(b - a) / pm.FPS for a, b, _ in pm.possession_spells(filled, runs)]
    transitions = pm.transitions(filled, runs)
    return {
        "raw_counts": {label: int((raw == i).sum()) for i, label in enumerate(pm.LABELS)},
        "raw_team_frames_with_ball_missing": int((~visible & np.isin(raw, [pm.H, pm.A])).sum()),
        "effective_team_spells": len(durations),
        "median_spell_s": float(np.median(durations)) if durations else None,
        "p90_spell_s": float(np.quantile(durations, 0.9)) if durations else None,
        "longest_spell_s": max(durations, default=None),
        "effective_transitions": len(transitions),
        "effective_direct_team_switches": sum(
            left in (pm.H, pm.A) and right in (pm.H, pm.A)
            for _, left, right in transitions
        ),
        "effective_D_frames": int((filled[frames] == pm.D).sum()),
    }
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE,
                        help="Optional teammate CSV for the structural comparison")
    parser.add_argument("--no-pre-kickoff", action="store_true",
                        help="Do not label the second-half stretch before kickoff as D")
    parser.add_argument("--output", type=Path, default=RESULTS / "possession_hysteresis_d.csv",
                        help="A new CSV path inside the repository results directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix.lower() != ".csv" or not output.is_relative_to(RESULTS.resolve()):
        parser.error("--output must be a CSV inside the ignored results directory")
    if output.exists():
        parser.error("--output already exists; choose a new CSV path")
 
    runs = [(start, end + 1) for _, start, end in half_video_blocks(GAME)]
    frames = np.concatenate([np.arange(start, end) for start, end in runs])
    distance, relative, visible, outside = load_tracking(frames)
    frozen = read_codes(FROZEN)
    baseline = read_codes(args.baseline)
    with np.load(ROOT / "cache/tactical_atd.npz", allow_pickle=False) as atd:
        dead_full = build_dead_mask(
            atd["players"], atd["ball"],
            tagging_path=asset_paths()["smart_tagging"],
            pre_kickoff_dead=not args.no_pre_kickoff,
        )
    dead = dead_full[frames]
    votes = team_votes(distance, relative, visible, outside)
    restart = nearest_votes(distance, visible, outside)
    challenger = np.full(source_frames(), pm.X, dtype=np.int8)
    challenger[frames] = decode(frames, votes, outside, dead, restart)
    if not np.array_equal(challenger[frames] == pm.D, dead):
        raise ValueError("Challenger D labels differ from the dead-ball mask")
 
    report = {
        "baseline_file": args.baseline.name,
        "hysteresis": describe(challenger, frames, runs, visible),
        "frozen_v1": describe(frozen, frames, runs, visible),
        "nearest_player_baseline": describe(baseline, frames, runs, visible),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(np.asarray(pm.LABELS)[challenger]))
    print(f"Saved {output}")
    print(json.dumps(report, indent=2, allow_nan=False))
 
 
if __name__ == "__main__":
    main()