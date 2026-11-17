
import csv
from pathlib import Path

import numpy as np

TOTAL_FRAMES = 199319

GRADED_WINDOWS = [
    (11185, 90651),
    (110808, 190197),
]

ALLOWED_LABELS = {"H", "A", "D", "X"}

SOURCE = Path("cache/decoded_states_v1.npz")
OUTPUT = Path("results/possession_v1.csv")


def build_submission():
    with np.load(SOURCE) as data:
        frames = data["frame"].astype(int)
        predictions = data["label"]

    # Verify the exact graded frame indices.
    expected_frames = np.concatenate([
        np.arange(start, end + 1)
        for start, end in GRADED_WINDOWS
    ])

    assert np.array_equal(frames, expected_frames), (
        "Model frames do not match the graded windows."
    )

    assert np.all(
        (frames >= 0) & (frames < TOTAL_FRAMES)
    )

    assert set(np.unique(predictions)).issubset(
        ALLOWED_LABELS
    )

    # Reconstruct the full video timeline.
    # Outside graded windows, abstain.
    labels = np.full(
        TOTAL_FRAMES, "X", dtype="<U1"
    )

    labels[frames] = predictions

    # Ensure that the reconstruction is lossless.
    assert np.array_equal(
        labels[frames], predictions
    )

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "label"])

        writer.writerows(
            (i, label)
            for i, label in enumerate(labels)
        )

    return labels


def verify_submission(labels):
    with OUTPUT.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.reader(file)

        assert next(reader) == ["frame", "label"]

        count = 0

        for expected_frame, row in enumerate(reader):
            assert len(row) == 2

            frame, label = row

            assert int(frame) == expected_frame
            assert label in ALLOWED_LABELS
            assert label == labels[expected_frame]

            count += 1

    assert count == TOTAL_FRAMES

    print("Submission format verified.")
    print("Frame count:", count)
    print("First frame:", 0)
    print("Last frame:", TOTAL_FRAMES - 1)

    print("\nFull-video label counts:")
    for state in ["H", "A", "D", "X"]:
        print(
            f"{state}: {int(np.sum(labels == state)):,}"
        )

    print("\nGraded-window label counts:")
    mask = np.zeros(TOTAL_FRAMES, dtype=bool)

    for start, end in GRADED_WINDOWS:
        mask[start:end + 1] = True

    for state in ["H", "A", "D", "X"]:
        print(
            f"{state}: "
            f"{int(np.sum(labels[mask] == state)):,}"
        )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    labels = build_submission()
    verify_submission(labels)
