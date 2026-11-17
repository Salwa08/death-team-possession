
from pathlib import Path
import numpy as np

# Provisional policy assumptions, not GT-tuned.
TEAM_CONFIDENCE = 0.80
DEAD_CONFIDENCE = 0.70
MAX_EVIDENCE_AGE = 25  # frames = 1 second


def evidence_distance(frames, observed):
    """Distance to nearest observed frame, respecting breaks."""
    n = len(frames)
    distance = np.full(n, n + 1, dtype=int)

    # Forward pass.
    for i in range(n):
        if observed[i]:
            distance[i] = 0
        elif i > 0 and frames[i] == frames[i - 1] + 1:
            distance[i] = min(distance[i], distance[i - 1] + 1)

    # Backward pass: offline inference can use future evidence.
    for i in range(n - 2, -1, -1):
        if frames[i + 1] == frames[i] + 1:
            distance[i] = min(distance[i], distance[i + 1] + 1)

    return distance


def decode(frames, posterior, h, a, outside):
    n = len(frames)
    labels = np.full(n, "X", dtype="<U1")

    observed = (
        np.isfinite(h)
        & np.isfinite(a)
        & ~outside
    )

    age = evidence_distance(frames, observed)
    recent = age <= MAX_EVIDENCE_AGE

    best = np.argmax(posterior, axis=1)
    confidence = np.max(posterior, axis=1)

    labels[
        (best == 0)
        & (confidence >= TEAM_CONFIDENCE)
        & recent
        & ~outside
    ] = "H"

    labels[
        (best == 1)
        & (confidence >= TEAM_CONFIDENCE)
        & recent
        & ~outside
    ] = "A"

    # Do not equate missing ball observations with D.
    labels[
        (best == 2)
        & (confidence >= DEAD_CONFIDENCE)
        & outside
    ] = "D"

    return labels


def spell_lengths(frames, labels, state):
    lengths = []
    start = None

    for i in range(len(labels)):
        if labels[i] != state:
            if start is not None:
                lengths.append(i - start)
                start = None
            continue

        if start is None:
            start = i
        elif frames[i] != frames[i - 1] + 1:
            lengths.append(i - start)
            start = i

    if start is not None:
        lengths.append(len(labels) - start)

    return np.asarray(lengths, dtype=int)


# Synthetic tests.
test_frames = np.array([0, 1, 2, 100, 101])
test_observed = np.array([True, False, False, False, True])

dist = evidence_distance(test_frames, test_observed)
assert dist.tolist() == [0, 1, 2, 1, 0]

test_p = np.array([
    [.95, .03, .02],
    [.95, .03, .02],
    [.95, .03, .02],
    [.05, .05, .90],
    [.05, .05, .90],
])

test_h = np.array([0.5, np.nan, np.nan, np.nan, np.nan])
test_a = np.array([5.0, np.nan, np.nan, np.nan, np.nan])
test_outside = np.array([False, False, False, True, True])

test_labels = decode(
    test_frames, test_p, test_h, test_a, test_outside
)

assert test_labels.tolist() == ["H", "H", "H", "D", "D"]
assert spell_lengths(
    test_frames, test_labels, "H"
).tolist() == [3]

print("Synthetic decoder tests passed.")


# Real timeline.
with np.load("cache/three_state_belief.npz") as d:
    frames = d["frame"]
    posterior = np.column_stack([
        d["belief_h"],
        d["belief_a"],
        d["belief_d"],
    ])

with np.load("cache/control_evidence.npz") as d:
    assert np.array_equal(frames, d["frame"])
    h = d["h_distance"]
    a = d["a_distance"]

with np.load("cache/game_state_evidence.npz") as d:
    assert np.array_equal(frames, d["frame"])
    outside = d["ball_outside_pitch"]

labels = decode(frames, posterior, h, a, outside)

output = Path("cache/decoded_states_v0.npz")
output.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output,
    frame=frames,
    label=labels,
)

print("\n=== FIRST DECODED TIMELINE ===")
print("Frames:", len(frames))

for state in ["H", "A", "D", "X"]:
    count = int(np.sum(labels == state))
    print(
        f"{state}: {count:,} frames "
        f"({100 * count / len(labels):.2f}%)"
    )

print("\n=== SPELL STATISTICS ===")
for state in ["H", "A", "D"]:
    lengths = spell_lengths(frames, labels, state)

    if len(lengths):
        print(
            f"{state}: {len(lengths)} spells, "
            f"median {np.median(lengths) / 25:.2f}s, "
            f"p90 {np.quantile(lengths, .9) / 25:.2f}s"
        )

print("\nOutside-pitch frames decoded D:",
      int(np.sum(outside & (labels == "D"))))
print("Saved:", output)
