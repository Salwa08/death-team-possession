
from pathlib import Path
import numpy as np

# States: H=0, A=1, D=2
# Prototype assumptions, NOT calibrated probabilities.
PRIOR = np.array([0.475, 0.475, 0.05])

TRANSITION = np.array([
    [0.996, 0.003, 0.001],
    [0.003, 0.996, 0.001],
    [0.010, 0.010, 0.980],
])


def make_emissions(h, a, hv, av, outside):
    """Build positive, relative observation likelihoods."""
    n = len(h)
    valid = np.isfinite(h) & np.isfinite(a)

    def control_score(distance, motion):
        result = np.zeros(n)

        motion_factor = np.zeros(n)
        good_motion = np.isfinite(motion)

        motion_factor[good_motion] = np.exp(
            -0.5 * (motion[good_motion] / 5.0) ** 2
        )

        result[valid] = (
            np.exp(-0.5 * (distance[valid] / 2.0) ** 2)
            * (0.5 + 0.5 * motion_factor[valid])
        )

        # Do not interpret out-of-pitch ball proximity
        # as evidence of active team possession.
        result[outside] = 0.0
        return result

    hs = control_score(h, hv)
    aa = control_score(a, av)

    # Small per-frame evidence increments.
    # These are explicit modeling hypotheses.
    log_emissions = np.column_stack([
        0.12 * hs,
        0.12 * aa,
        0.22 * outside.astype(float)
        - 0.08 * np.maximum(hs, aa),
    ])

    # Stabilize exponentiation.
    log_emissions -= log_emissions.max(
        axis=1, keepdims=True
    )

    return np.exp(log_emissions)


def forward_backward(frames, emissions):
    n = len(frames)
    forward = np.empty((n, 3))

    for i in range(n):
        if i == 0 or frames[i] != frames[i - 1] + 1:
            predicted = PRIOR
        else:
            predicted = forward[i - 1] @ TRANSITION

        unnormalized = predicted * emissions[i]
        forward[i] = (
            unnormalized / unnormalized.sum()
        )

    posterior = np.empty_like(forward)
    backward = np.ones(3)

    for i in range(n - 1, -1, -1):
        joint = forward[i] * backward
        posterior[i] = joint / joint.sum()

        if i > 0:
            if frames[i] != frames[i - 1] + 1:
                backward = np.ones(3)
            else:
                backward = TRANSITION @ (
                    emissions[i] * backward
                )
                backward /= backward.sum()

    return posterior


# Synthetic test:
# Barcelona control -> ball outside -> no observation
n = 200
frames = np.arange(n)

h = np.full(n, np.nan)
a = np.full(n, np.nan)
hv = np.full(n, np.nan)
av = np.full(n, np.nan)
outside = np.zeros(n, dtype=bool)

h[:75], a[:75] = 0.5, 7.0
hv[:75], av[:75] = 0.5, 4.0
outside[75:125] = True

emissions = make_emissions(
    h, a, hv, av, outside
)
posterior = forward_backward(frames, emissions)

assert posterior[40, 0] > 0.7
assert posterior[100, 2] > 0.7
assert np.allclose(posterior.sum(axis=1), 1.0)

# Missing observations must not automatically
# count as dead-ball evidence.
neutral = make_emissions(
    np.full(50, np.nan),
    np.full(50, np.nan),
    np.full(50, np.nan),
    np.full(50, np.nan),
    np.zeros(50, dtype=bool),
)
neutral_p = forward_backward(np.arange(50), neutral)
assert neutral_p[25, 2] < 0.2

# No future evidence may cross a frame discontinuity.
broken_frames = np.r_[
    np.arange(20),
    np.arange(100, 120),
]
broken_outside = np.r_[
    np.zeros(20, dtype=bool),
    np.ones(20, dtype=bool),
]
nan40 = np.full(40, np.nan)
broken_e = make_emissions(
    nan40, nan40, nan40, nan40, broken_outside
)
broken_p = forward_backward(
    broken_frames, broken_e
)
assert np.isclose(broken_p[19, 2], PRIOR[2], atol=0.01)

print("Synthetic three-state tests passed.")
print(
    "Synthetic H, A, D at outside interval midpoint:",
    np.round(posterior[100], 3),
)


# Real tracking
with np.load("cache/control_evidence.npz") as d:
    real_frames = d["frame"]
    h = d["h_distance"]
    a = d["a_distance"]
    hv = d["h_relative_speed"]
    av = d["a_relative_speed"]

with np.load("cache/game_state_evidence.npz") as d:
    assert np.array_equal(real_frames, d["frame"])
    outside = d["ball_outside_pitch"]

emissions = make_emissions(h, a, hv, av, outside)
posterior = forward_backward(real_frames, emissions)

output = Path("cache/three_state_belief.npz")
output.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output,
    frame=real_frames,
    belief_h=posterior[:, 0],
    belief_a=posterior[:, 1],
    belief_d=posterior[:, 2],
)

print("\nReal tracking:")
print("Frames:", len(real_frames))
print("Mean H/A/D belief:",
      np.round(posterior.mean(axis=0), 3))
print("D belief quantiles:",
      np.round(
          np.quantile(posterior[:, 2], [.1, .5, .9]),
          3
      ))
print(
    "Frames with D belief > 0.7:",
    int(np.sum(posterior[:, 2] > 0.7))
)
print(
    "Median D belief when ball outside:",
    round(float(np.median(posterior[outside, 2])), 3)
    if outside.any() else "N/A"
)
print(
    "Median D belief when ball not outside:",
    round(float(np.median(posterior[~outside, 2])), 3)
)
print("Saved:", output)
