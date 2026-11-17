
from pathlib import Path
import numpy as np

from experiments.control_evidence import extract_evidence

SOURCE = Path("data/tactical/BAR-ATH_tactical_atd.txt")
OUTPUT = Path("cache/control_evidence.npz")
PITCH = np.array([105.3, 68.0])

# Half-open intervals; frame indices are zero-based.
INTERVALS = [
    (11185, 90652),
    (110808, 190198),
]

EXPECTED_FRAMES = sum(b - a for a, b in INTERVALS)

def graded(frame):
    return any(a <= frame < b for a, b in INTERVALS)

def parse(line):
    parts = line.strip().split(":")
    if len(parts) != 3:
        return None

    frame = int(parts[0]) - 1

    groups = [
        g.split(",") for g in parts[1].split(";")
    ]

    if len(groups) != 46:
        raise ValueError(
            f"Expected 46 slots at frame {frame}"
        )

    players = np.array([
        [float(g[0]), float(g[1])]
        for g in groups
    ]) * PITCH

    ball = np.array([
        float(v) for v in parts[2].split(",")[:2]
    ]) * PITCH

    return frame, players, ball


rows = []
previous = None

with SOURCE.open() as f:
    for line in f:
        parsed = parse(line)
        if parsed is None:
            continue

        frame, players, ball = parsed

        if previous is not None and previous[0] == frame - 1:
            _, prev_players, prev_ball = previous
        else:
            prev_players = np.full((46, 2), np.nan)
            prev_ball = np.full(2, np.nan)

        if graded(frame):
            e = extract_evidence(
                prev_players,
                players,
                prev_ball,
                ball,
            )

            rows.append([
                frame,
                e["H"]["distance"],
                e["A"]["distance"],
                e["H"]["relative_speed"],
                e["A"]["relative_speed"],
                e["distance_margin_H"],
                e["H"]["nearest_slot"],
                e["A"]["nearest_slot"],
            ])

        previous = (frame, players, ball)


data = np.asarray(rows, dtype=float)

assert len(data) == EXPECTED_FRAMES, (
    f"Expected {EXPECTED_FRAMES}, got {len(data)}"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT,
    frame=data[:, 0].astype(np.int32),
    h_distance=data[:, 1],
    a_distance=data[:, 2],
    h_relative_speed=data[:, 3],
    a_relative_speed=data[:, 4],
    distance_margin_h=data[:, 5],
    h_nearest_slot=data[:, 6].astype(np.int32),
    a_nearest_slot=data[:, 7].astype(np.int32),
)

print("Saved:", OUTPUT)
print("Graded frames:", len(data))

names = [
    "H distance",
    "A distance",
    "H relative speed",
    "A relative speed",
    "Distance margin",
]

for i, name in enumerate(names, start=1):
    count = np.isfinite(data[:, i]).sum()
    print(
        f"{name}: {count:,} valid "
        f"({100 * count / len(data):.2f}%)"
    )

print("Evidence timeline ready.")
