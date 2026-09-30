# ROBOT2026 Camera-Ready Experiment Handoff

This document summarizes the additional experiments performed after the
ROBOT2026 reviews, where to find the results, the main conclusions, and the
recommended next steps for revising the paper.

## Repository State

- Preserved original submitted/presented state:
  - Branch: `1st_results_paper`
  - Tag: `robot2026-submitted-v1`
- Camera-ready experiment branch:
  - Branch: `robot2026-camera-ready`
- Original first-paper result artifacts:
  - `outputs_best/`
- New camera-ready raw training artifacts:
  - `outputs_camera_ready/`
- New camera-ready summaries and reports:
  - `reports/robot2026_camera_ready/s2_ablation/`

The original submitted results were not overwritten. All new experiments were
run in separate `outputs_camera_ready/` folders.

## What Was Done

The reviewer-critical issue was the lack of controlled training-time ablations
for S2 stair climbing. A new controlled S2 ablation campaign was run with five
training seeds per arm:

1. `s2_base`
   - Original S2 setup.
   - Terrain-relative termination.
   - Original explicit stair-shaping terms.
   - Seeds: `42, 43, 44, 45, 46`
   - Seed 42 is the preserved published run from `outputs_best/2025-12-06/17-36-50`.
   - Seeds 43-46 are new camera-ready runs.

2. `s2_world_frame_termination`
   - Same S2 task.
   - Health check changed to world-frame termination:
     `env.make_kwargs.check_healthy_z_relative=false`
   - Seeds: `42, 43, 44, 45, 46`

3. `s2_no_shaping`
   - Same S2 task.
   - Explicit stair shaping disabled:
     `env.make_kwargs.height_reward_weight=0`
     `env.make_kwargs.step_bonus=0`
   - Seeds: `42, 43, 44, 45, 46`

Each run used:

- PPO training.
- `30,000,000` timesteps.
- Existing S2 config: `env=humanoid_stairs_easy`.
- Best checkpoint selected during training by mean reward over 5 deterministic
  evaluation episodes, using the existing `EvalCallback` protocol.

After training, all 15 runs were re-evaluated with:

- 100 deterministic episodes per run.
- Evaluation seed: `123`.
- Success criterion: `max_step_reached >= num_steps`.

## Where The Results Are

Experiment plan and manifest:

- `experiments/robot2026_camera_ready/README.md`
- `experiments/robot2026_camera_ready/s2_ablation_manifest.csv`
- `experiments/robot2026_camera_ready/run_s2_ablation_training.sh`

Evaluation/report script:

- `scripts/evaluate/evaluate_robot2026_s2_ablation.py`

Final report and CSV outputs:

- `reports/robot2026_camera_ready/s2_ablation/after_action_report.md`
- `reports/robot2026_camera_ready/s2_ablation/arm_summary.csv`
- `reports/robot2026_camera_ready/s2_ablation/seed_summary.csv`
- `reports/robot2026_camera_ready/s2_ablation/per_episode.csv`

Success-based checkpoint diagnostic:

- `reports/robot2026_camera_ready/s2_success_checkpoint_diagnostic/after_action_report.md`
- `reports/robot2026_camera_ready/s2_success_checkpoint_diagnostic/arm_summary.csv`
- `reports/robot2026_camera_ready/s2_success_checkpoint_diagnostic/final_success_selected_summary.csv`
- `reports/robot2026_camera_ready/s2_success_checkpoint_diagnostic/selected_checkpoints.csv`
- `reports/robot2026_camera_ready/s2_success_checkpoint_diagnostic/checkpoint_scan_summary.csv`

Raw trained models and logs:

- `outputs_camera_ready/s2_base/seed_43`
- `outputs_camera_ready/s2_base/seed_44`
- `outputs_camera_ready/s2_base/seed_45`
- `outputs_camera_ready/s2_base/seed_46`
- `outputs_camera_ready/s2_world_frame_termination/seed_42`
- `outputs_camera_ready/s2_world_frame_termination/seed_43`
- `outputs_camera_ready/s2_world_frame_termination/seed_44`
- `outputs_camera_ready/s2_world_frame_termination/seed_45`
- `outputs_camera_ready/s2_world_frame_termination/seed_46`
- `outputs_camera_ready/s2_no_shaping/seed_42`
- `outputs_camera_ready/s2_no_shaping/seed_43`
- `outputs_camera_ready/s2_no_shaping/seed_44`
- `outputs_camera_ready/s2_no_shaping/seed_45`
- `outputs_camera_ready/s2_no_shaping/seed_46`

The preserved published S2 seed 42 is in:

- `outputs_best/2025-12-06/17-36-50`

## Main Results

Mean success rate across five seeds:

| Arm | Seeds | Mean success | S.D. |
|---|---:|---:|---:|
| `s2_base` | 5 | 28.8% | 40.5 pp |
| `s2_world_frame_termination` | 5 | 35.0% | 32.3 pp |
| `s2_no_shaping` | 5 | 81.2% | 8.3 pp |

Per-seed success rates:

| Arm | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---:|---:|---:|---:|---:|
| `s2_base` | 92% | 47% | 0% | 0% | 5% |
| `s2_world_frame_termination` | 13% | 0% | 25% | 76% | 61% |
| `s2_no_shaping` | 83% | 94% | 76% | 72% | 81% |

## Conclusions

The new results substantially change the paper story.

1. The original S2 seed 42 result remains strong, but it is not representative
   of the five-seed `s2_base` distribution. The base arm showed very high
   seed-to-seed variability.

2. The controlled ablation does not support a strong claim that world-frame
   termination categorically prevents S2 learning. The world-frame arm was
   mixed: some seeds performed poorly, but others reached 61-76% success.

3. The controlled ablation does not support a strong claim that explicit
   height-progress and step-milestone reward shaping enabled S2 success. The
   `s2_no_shaping` arm performed best and most consistently.

4. The results reveal a likely checkpoint-selection issue. Some policies
   achieved high reward and long episode length while failing to climb the
   stairs. For example, `s2_base` seed 44 had:
   - 0% success
   - mean episode length 983.8 steps
   - mean reward 14758.9
   - mean maximum step reached only 1.98

   This suggests the reward-based checkpoint criterion can select long-surviving
   non-climbing policies.

5. The camera-ready paper should therefore become more modest and more
   transparent. It should no longer frame S2 as clear causal evidence that
   reward shaping and terrain-relative termination enabled stair learning.

## Checkpoint-Selection Diagnostic

The suggested diagnostic next step was executed after the first ablation report.
It asked whether success-based checkpoint selection changes the conclusion.

Protocol:

- Coarse scan: every 1M checkpoint, 10 deterministic episodes.
- Refinement: checkpoints within +/-1M of the coarse winner, 20 deterministic
  episodes.
- Final evaluation: selected success-best checkpoint, 100 deterministic
  episodes.
- Evaluation seed: `123`.
- Checkpoints scanned: 561 candidate evaluations across all runs.

Mean success rate comparison:

| Arm | Reward-selected | Success-selected diagnostic |
|---|---:|---:|
| `s2_base` | 28.8% | 42.8% |
| `s2_world_frame_termination` | 35.0% | 51.2% |
| `s2_no_shaping` | 81.2% | 86.6% |

Interpretation:

- Reward-based checkpoint selection was a real confound.
- Success-selected checkpoints improve `s2_base` and
  `s2_world_frame_termination`, so some useful stair-completion policies existed
  but were not selected by reward.
- The confound does not fully explain the result. `s2_no_shaping` remains the
  strongest and most stable arm after success-based selection.
- The paper should report checkpoint-selection mismatch as a methodological
  lesson, not as a reason to restore the original strong reward-shaping claim.

## Recommended Paper Changes

1. Replace single-seed S2 claims with the five-seed S2 ablation table.

2. Remove or heavily soften claims such as:
   - reward shaping enabled S2 stair climbing
   - terrain-relative termination was the single change that enabled learning
   - world-frame termination prevents learning

3. Reframe the contribution as a reproducible practitioner-oriented case study
   showing how reward design, termination logic, checkpoint selection, and seed
   variability interact in humanoid stair learning.

4. Explicitly describe checkpoint selection:
   - best checkpoint selected during training by mean reward
   - 5 deterministic evaluation episodes during training
   - final reported numbers re-evaluated separately on 100 deterministic
     episodes with evaluation seed 123

5. Add a limitation about reward/checkpoint mismatch:
   - high healthy reward and long survival can produce high reward without
     stair completion
   - success-rate-based checkpoint selection may be more appropriate for S2

6. Keep terrain-relative termination as a practical recommendation, but phrase
   it cautiously:
   - it avoids a known elevated-terrain failure mode
   - it did not produce decisive improvement in the five-seed S2 training
     ablation

7. Treat explicit stair shaping as unsupported by this controlled S2 ablation.
   If discussed, state that the tested shaping terms did not improve S2
   completion under the current protocol.

## Suggested Next Steps

1. Use the success-based checkpoint diagnostic as additional evidence that
   checkpoint selection was a confound. Do not replace the reward-selected
   report; present the two protocols clearly.

2. Revise the manuscript around the current five-seed findings. Avoid more
   training unless a reviewer-response decision specifically requires it.

3. Update the manuscript tables and discussion using:
   - `arm_summary.csv`
   - `seed_summary.csv`
   - `after_action_report.md`
   from the reward-selected report, plus the success-selected diagnostic report
   if space allows.

4. Add the camera-ready evidence as additional experiments, not as replacement
   for the original submitted artifacts. The original artifacts remain
   preserved on `1st_results_paper`.

## Useful Commands

Regenerate the final evaluation report:

```bash
.venv/bin/python scripts/evaluate/evaluate_robot2026_s2_ablation.py \
  --manifest experiments/robot2026_camera_ready/s2_ablation_manifest.csv \
  --episodes 100 \
  --seed 123 \
  --deterministic \
  --strict \
  --output-dir reports/robot2026_camera_ready/s2_ablation
```

Inspect the arm summary:

```bash
cat reports/robot2026_camera_ready/s2_ablation/arm_summary.csv
```

Inspect the seed summary:

```bash
cat reports/robot2026_camera_ready/s2_ablation/seed_summary.csv
```
