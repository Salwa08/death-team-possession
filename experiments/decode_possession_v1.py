
from pathlib import Path
import numpy as np

TEAM_CONFIDENCE = 0.80
DEAD_CONFIDENCE = 0.70


def decode_without_age_gate(posterior, outside):
    labels = np.full(
        len(posterior), "X", dtype="<U1"
    )

    best = np.argmax(posterior, axis=1)
    confidence = posterior.max(axis=1)

    active = ~outside

    labels[
        active
        & (best == 0)
        & (confidence >= TEAM_CONFIDENCE)
    ] = "H"

    labels[
        active
        & (best == 1)
        & (confidence >= TEAM_CONFIDENCE)
    ] = "A"

    labels[
        outside
        & (best == 2)
        & (confidence >= DEAD_CONFIDENCE)
    ] = "D"

    return labels


with np.load("cache/decoded_states_v0.npz") as d:
    frames = d["frame"]
    old_labels = d["label"]

with np.load("cache/three_state_belief.npz") as d:
    assert np.array_equal(frames, d["frame"])

    posterior = np.column_stack([
        d["belief_h"],
        d["belief_a"],
        d["belief_d"],
    ])

with np.load("cache/game_state_evidence.npz") as d:
    assert np.array_equal(frames, d["frame"])
    outside = d["ball_outside_pitch"]

new_labels = decode_without_age_gate(
    posterior, outside
)

# Removing the age gate must not erase or
# reverse previously accepted labels.
assert np.all(
    new_labels[old_labels != "X"]
    == old_labels[old_labels != "X"]
)

# Outside-pitch observations cannot produce H/A.
assert not np.any(
    outside & np.isin(new_labels, ["H", "A"])
)

print("Decoder invariance tests passed.")

print("\n=== DECODER COMPARISON ===")
print(f"{'State':<8} {'v0':>10} {'v1':>10} {'Change':>10}")

for state in ["H", "A", "D", "X"]:
    old = int(np.sum(old_labels == state))
    new = int(np.sum(new_labels == state))

    print(
        f"{state:<8} {old:>10,} "
        f"{new:>10,} {new-old:>+10,}"
    )

newly_assigned = (
    (old_labels == "X")
    & np.isin(new_labels, ["H", "A"])
)

print("\nNewly assigned team frames:",
      int(newly_assigned.sum()))

print(
    "Remaining X percentage:",
    round(
        100 * np.mean(new_labels == "X"), 2
    )
)

# Report how far the newly assigned frames
# are from actual player-ball observations.
with np.load("cache/control_evidence.npz") as d:
    assert np.array_equal(frames, d["frame"])
    observed = (
        np.isfinite(d["h_distance"])
        & np.isfinite(d["a_distance"])
        & ~outside
    )

distance = np.full(len(frames), np.inf)

for i in range(len(frames)):
    if observed[i]:
        distance[i] = 0
    elif i > 0 and frames[i] == frames[i-1] + 1:
        distance[i] = distance[i-1] + 1

for i in range(len(frames)-2, -1, -1):
    if frames[i+1] == frames[i] + 1:
        distance[i] = min(
            distance[i], distance[i+1] + 1
        )

if newly_assigned.any():
    ages = distance[newly_assigned] / 25

    print("\n=== NEW ASSIGNMENT DIAGNOSTICS ===")
    print(
        "Median distance to observation:",
        round(float(np.median(ages)), 2), "s"
    )
    print(
        "90th percentile distance:",
        round(float(np.quantile(ages, .9)), 2), "s"
    )
    print(
        "New assignments > 5s from observation:",
        int(np.sum(ages > 5))
    )
    print(
        "New assignments > 10s from observation:",
        int(np.sum(ages > 10))
    )

output = Path("cache/decoded_states_v1.npz")
np.savez_compressed(
    output,
    frame=frames,
    label=new_labels,
)

print("\nSaved:", output)
