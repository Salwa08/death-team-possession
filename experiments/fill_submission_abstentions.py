"""Export a copy of a submission with scorer-filled labels in the two graded halves."""

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks, source_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    table = pm.read_label_csv(args.source)
    reason = pm.validate_submission(table, source_frames())
    if reason:
        parser.error(f"{args.source}: {reason}")

    raw = pm.label_codes(table)
    runs = [(start, end + 1) for _, start, end in half_video_blocks(GAME)]
    graded = np.zeros(len(raw), dtype=bool)
    for start, stop in runs:
        if np.all(raw[start:stop] == pm.X):
            parser.error("a graded half has no H, A, or D label to carry")
        graded[start:stop] = True
    filled = pm.fill_abstentions(raw, runs)

    try:
        with args.destination.open("x", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("frame", "label"))
            writer.writerows(enumerate(np.asarray(pm.LABELS)[filled]))
    except FileExistsError:
        parser.error(f"destination already exists: {args.destination}")

    print(f"Scorer-effective team spells: {len(pm.possession_spells(filled, runs)):,}")
    print(f"Graded X frames: {np.count_nonzero(raw[graded] == pm.X):,} -> "
          f"{np.count_nonzero(filled[graded] == pm.X):,}")
    counts = ", ".join(f"{label} {np.count_nonzero(filled[graded] == code):,}"
                       for code, label in enumerate(pm.LABELS))
    print(f"Graded label counts: {counts}")
    print(f"Saved {args.destination}")


if __name__ == "__main__":
    main()
