# ROBOT2026 Camera-Ready Experiments

This folder defines the additional controlled experiments for the ROBOT2026
camera-ready revision. The original submitted/presented results are preserved on
the `1st_results_paper` branch and in `outputs_best/`; do not overwrite them.

## Scope

The reviewer-critical evidence is concentrated on S2 stair climbing:

- `s2_base`: original S2 formulation with terrain-relative termination.
- `s2_world_frame_termination`: same S2 task, but health is checked in the
  world frame.
- `s2_no_shaping`: same S2 task, but height progress and step milestone shaping
  are disabled.

S4 world-frame termination is intentionally excluded from the primary new
training plan because the current S4 stairs rise only 0.50 m, so the fixed
upper health bound is not expected to trigger. S4 changes should be textual
clarifications unless a new, pre-registered S4 experiment is explicitly added.

## Output Boundaries

- Raw training outputs: `outputs_camera_ready/<arm>/seed_<seed>/`
- Generated summaries: `reports/robot2026_camera_ready/s2_ablation/`
- Preserved first paper outputs: `outputs_best/`

The `outputs_camera_ready/` folder is ignored by Git. Keep generated CSV and
Markdown summaries under `reports/robot2026_camera_ready/` so conclusions are
tracked without committing large model artifacts.

## Run Plan

The new S2 runs are deterministic in folder naming:

```bash
bash experiments/robot2026_camera_ready/run_s2_ablation_training.sh
```

The published S2 seed 42 is read from:

```text
outputs_best/2025-12-06/17-36-50
```

New camera-ready runs are written to:

```text
outputs_camera_ready/s2_base/seed_43
outputs_camera_ready/s2_base/seed_44
outputs_camera_ready/s2_base/seed_45
outputs_camera_ready/s2_base/seed_46
outputs_camera_ready/s2_world_frame_termination/seed_42
outputs_camera_ready/s2_world_frame_termination/seed_43
outputs_camera_ready/s2_world_frame_termination/seed_44
outputs_camera_ready/s2_world_frame_termination/seed_45
outputs_camera_ready/s2_world_frame_termination/seed_46
outputs_camera_ready/s2_no_shaping/seed_42
outputs_camera_ready/s2_no_shaping/seed_43
outputs_camera_ready/s2_no_shaping/seed_44
outputs_camera_ready/s2_no_shaping/seed_45
outputs_camera_ready/s2_no_shaping/seed_46
```

## Evaluation

After any subset of runs completes, generate/refresh the after-action report:

```bash
python scripts/evaluate/evaluate_robot2026_s2_ablation.py \
  --manifest experiments/robot2026_camera_ready/s2_ablation_manifest.csv \
  --episodes 100 \
  --seed 123 \
  --deterministic \
  --output-dir reports/robot2026_camera_ready/s2_ablation
```

By default the evaluator skips missing runs, which makes it useful during a
long campaign. Add `--strict` when producing final camera-ready numbers.

## Reporting Rule

Report every pre-registered seed. Do not replace or rerun only failed seeds.
If a run crashes before completing, record it in the after-action report with
the failure mode and decide whether the whole arm needs to be restarted before
looking at final performance.
