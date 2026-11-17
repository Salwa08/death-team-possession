# Reproduce the v1 submission

Use Python 3.12 in an isolated local copy of the repository. The reference environment uses Python 3.12.4. Place the licensed tracking text at `data/tactical/BAR-ATH_tactical_atd.txt` relative to that copy's root. Start with empty `cache/` and `results/` directories.

Run these commands from the copy's root in order. The six cache producers also execute on import. The ball trajectory builder checks alignment against the control evidence cache, and the v1 decoder checks its output against the v0 cache.

```text
python -m pip install -r requirements-v1.txt
python -B -m experiments.build_evidence_timeline
python -B -m experiments.build_ball_trajectory
python -B -m experiments.game_state_evidence
python -B -m experiments.three_state_hmm
python -B -m experiments.decode_possession
python -B -m experiments.decode_possession_v1
python -B -m experiments.build_submission
python -B evaluation/validate_submission.py results/possession_v1.csv --rows 199319
```

The validator checks the `frame,label` CSV with 199,319 rows and frame IDs 0 through 199,318. Compare the generated file's SHA-256 with the frozen v1 candidate:

```text
8de50a08fca501a3f0bf5d94ce8f0aecb417729baffaa6d8e86e913363b0e0cd
```

In PowerShell, use `Get-FileHash -Algorithm SHA256 results/possession_v1.csv`.

## Source-only staging allowlist

```text
.gitignore
REPRODUCE_V1.md
requirements-v1.txt
experiments/build_evidence_timeline.py
experiments/control_evidence.py
experiments/build_ball_trajectory.py
experiments/game_state_evidence.py
experiments/three_state_hmm.py
experiments/decode_possession.py
experiments/decode_possession_v1.py
experiments/build_submission.py
```

The validator also needs the already tracked `evaluation/validate_submission.py`, `evaluation/possession_metrics.py`, and `games.py`. Keep licensed match assets, caches, and generated predictions local.
