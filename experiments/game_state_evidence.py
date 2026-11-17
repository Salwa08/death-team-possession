
from pathlib import Path
import numpy as np

SOURCE = Path("cache/ball_trajectory.npz")
OUTPUT = Path("cache/game_state_evidence.npz")

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

with np.load(SOURCE) as d:
    frames = d["frame"]
    ball = d["position"]
    visible = d["visible"]

# Geometry-based evidence only.
# No match labels or video annotations are used.
outside = (
    visible
    & (
        (ball[:, 0] < 0)
        | (ball[:, 0] > PITCH_LENGTH)
        | (ball[:, 1] < 0)
        | (ball[:, 1] > PITCH_WIDTH)
    )
)

# Do not mistake an unobserved ball for
# an observed ball outside the pitch.
assert not np.any(outside & ~visible)

# Find contiguous outside-pitch intervals.
starts = []
durations = []

start = None

for i in range(len(frames)):
    if outside[i]:
        if start is None:
            start = i
        elif frames[i] != frames[i - 1] + 1:
            starts.append(frames[start])
            durations.append(i - start)
            start = i
    elif start is not None:
        starts.append(frames[start])
        durations.append(i - start)
        start = None

if start is not None:
    starts.append(frames[start])
    durations.append(len(frames) - start)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT,
    frame=frames,
    ball_visible=visible,
    ball_outside_pitch=outside,
)

print("Graded frames:", len(frames))
print("Visible ball frames:", int(visible.sum()))
print("Outside-pitch frames:", int(outside.sum()))
print("Outside-pitch intervals:", len(durations))

if durations:
    d = np.asarray(durations)
    print("Median interval duration:",
          round(np.median(d) / 25, 3), "s")
    print("Longest interval:",
          round(np.max(d) / 25, 3), "s")

print("Saved:", OUTPUT)
