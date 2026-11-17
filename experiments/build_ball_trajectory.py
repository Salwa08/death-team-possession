
from pathlib import Path
import numpy as np

SOURCE = Path("data/tactical/BAR-ATH_tactical_atd.txt")
OUTPUT = Path("cache/ball_trajectory.npz")

FPS = 25
PITCH = np.array([105.3, 68.0])

INTERVALS = [
    (11185, 90652),
    (110808, 190198),
]


def graded(frame):
    return any(a <= frame < b for a, b in INTERVALS)


frames = []
positions = []

with SOURCE.open() as f:
    for line in f:
        parts = line.strip().split(":")
        if len(parts) != 3:
            continue

        frame = int(parts[0]) - 1

        if not graded(frame):
            continue

        ball = np.array([
            float(v)
            for v in parts[2].split(",")[:2]
        ]) * PITCH

        frames.append(frame)
        positions.append(ball)

frames = np.asarray(frames, dtype=np.int32)
positions = np.asarray(positions, dtype=float)

expected = sum(b - a for a, b in INTERVALS)
assert len(frames) == expected

# Velocity exists only when the ball is visible
# in two genuinely consecutive frames.
velocity = np.full_like(positions, np.nan)

consecutive = np.diff(frames) == 1
visible = np.isfinite(positions).all(axis=1)

valid_motion = (
    consecutive
    & visible[:-1]
    & visible[1:]
)

indices = np.flatnonzero(valid_motion) + 1

velocity[indices] = (
    positions[indices] - positions[indices - 1]
) * FPS

speed = np.linalg.norm(velocity, axis=1)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT,
    frame=frames,
    position=positions,
    velocity=velocity,
    speed=speed,
    visible=visible,
)

print("Frames:", len(frames))
print("Ball visible:", int(visible.sum()))
print("Ball velocity available:",
      int(np.isfinite(speed).sum()))

# Confirm alignment with our existing evidence.
with np.load("cache/control_evidence.npz") as data:
    assert np.array_equal(frames, data["frame"])

print("Timeline alignment verified.")
print("Saved:", OUTPUT)
