# Team possession from tracking and hysteresis

## Approach

We estimate home possession (H), away possession (A), and dead ball (D) from the tactical tracking data. For each frame, we find the nearest tracked slot on each team and measure its distance and velocity relative to the ball. The team tags on the slots determine which side they belong to. A slot can change player identity between frames.

A team earns a tracking vote when the ball is visible in pitch, the team’s nearest slot is within 3 m, the other team is at least 1 m farther away, and relative speed is at most 3 m/s. The decoder acquires a team after five consecutive votes and switches an established team after ten consecutive opposing votes. It retains the incumbent for up to 25 frames after its latest supporting vote. At the start of a half or after a stop, the first eligible in-pitch observation can assign the nearest team immediately.

## Dead ball and restarts

The dead-ball mask combines three tracking cues: the ball is visibly outside for at least 1 s, its net displacement over 0.96 s remains below 1 m for at least 3 s, or the ball is missing and the smoothed mean player speed is below 1.5 m/s for at least 5 s. The player-speed calculation excludes tracking slots 23 and 45. The mask closes gaps of at most 1 s and retains runs lasting at least 3 s.

The supplied Smart Tagging XML gives the start of the second-half kickoff annotation. The decoder treats the graded stretch before that start plus 88 frames as dead ball. The offset is a fixed convention in this version.

The decoder emits X where it cannot assign a team or dead ball. The final export applies the official scorer’s forward fill within each graded half, using the first decided label for leading X frames. The export retains X outside the graded windows.

## Inputs and limitations

The model uses the supplied tactical tracking TXT, tactical metadata XML, and automatic Smart Tagging XML. It has fixed decision rules and no training stage or labelled training inputs. Position differences supply relative motion. Local unit checks found that stored player speeds behave like m/s, despite the XML channel name and declared unit. The speed rule uses the stored player values directly.

Occlusion during slow live play can resemble a stoppage under these rules. A ball resting in pitch can also be in play. The kickoff offset can put the restart boundary a few frames late, and the first nearest-team assignment can be wrong when a defender is closer to the ball. Forward fill extends earlier decisions through periods with weak or missing tracking.

## Reproduction

Use Python 3.12 in a clean copy with the supplied assets under `data/tactical/`. The commands expect `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`, and `BAR-ATH_tactical_smart_tagging.xml`. Keep these assets local. Run from the repository root with empty local `cache/` and `results/` directories. If you use `MTAH_DATA_DIR`, point it to this copy’s `data` folder so all commands use the same inputs.

```text
python -m pip install -r requirements-v1.txt
python -B -m experiments.build_evidence_timeline
python -B -m experiments.build_ball_trajectory
python -B -m experiments.game_state_evidence
python -B -m experiments.three_state_hmm
python -B -m experiments.decode_possession
python -B -m experiments.decode_possession_v1
python -B -m experiments.build_submission
python -B -m experiments.reconstruct_tactical_atd
python -B starter/baseline.py --window 50 --out results/reference
python -B -m experiments.possession_hysteresis_d --baseline results/reference/baseline_smooth50.csv --output results/possession_hysteresis_d_rerun.csv
python -B -m experiments.fill_submission_abstentions results/possession_hysteresis_d_rerun.csv results/final_filled.csv
python -B evaluation/validate_submission.py results/final_filled.csv --rows 199319
```

The HMM v1 CSV and the organizer baseline feed the decoder’s printed comparison report. They play no role in the final labels. The final CSV has 199,319 rows and SHA-256 `e29e06182607ec853030cb50599dd3d7e9a25717b7825ea744c6e10696ea8631`.
