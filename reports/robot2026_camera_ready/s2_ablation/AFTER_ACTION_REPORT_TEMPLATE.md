# S2 Ablation After-Action Report

## Experiment

- Branch: `robot2026-camera-ready`
- Preserved first-paper branch: `1st_results_paper`
- Manifest: `experiments/robot2026_camera_ready/s2_ablation_manifest.csv`
- Raw output root: `outputs_camera_ready/`
- Published seed 42 source: `outputs_best/2025-12-06/17-36-50`

## Arms

- `s2_base`: S2 with terrain-relative termination and original shaping.
- `s2_world_frame_termination`: S2 with world-frame health bounds during
  training.
- `s2_no_shaping`: S2 with `height_reward_weight=0` and `step_bonus=0`.

## Pre-Registered Success Criterion

An evaluation episode is counted as successful when the agent reaches all S2
steps, i.e. `max_step_reached >= num_steps` from the run config.

## Run Status

| Arm | Seeds | Status | Notes |
|---|---:|---|---|
| s2_base | 42-46 | pending | seed 42 already exists from first-paper results |
| s2_world_frame_termination | 42-46 | pending | new runs |
| s2_no_shaping | 42-46 | pending | new runs |

## Results Summary

Replace this section with the generated `after_action_report.md` after running:

```bash
python scripts/evaluate/evaluate_robot2026_s2_ablation.py \
  --manifest experiments/robot2026_camera_ready/s2_ablation_manifest.csv \
  --episodes 100 \
  --seed 123 \
  --deterministic \
  --output-dir reports/robot2026_camera_ready/s2_ablation
```

## Conclusions

- Pending.

## Paper Changes Supported

- Pending.
