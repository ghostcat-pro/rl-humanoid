#!/usr/bin/env bash
set -euo pipefail

# ROBOT2026 camera-ready S2 ablation campaign.
#
# Run from the repository root, ideally on the same DEEPLAB environment used for
# the first paper results. Raw outputs go to outputs_camera_ready/ and never
# overwrite outputs_best/.

COMMON_S2=(
  env=humanoid_stairs_easy
  training.total_timesteps=30000000
)

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl_humanoid_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

run_training() {
  local arm="$1"
  local seed="$2"
  shift 2

  "${PYTHON_BIN}" scripts/train/train_sb3.py \
    "${COMMON_S2[@]}" \
    seed="${seed}" \
    hydra.run.dir="outputs_camera_ready/${arm}/seed_${seed}" \
    "$@"
}

# A: S2 base. Seed 42 is the published run in outputs_best/2025-12-06/17-36-50.
for seed in 43 44 45 46; do
  run_training s2_base "${seed}"
done

# B: S2 with world-frame termination.
for seed in 42 43 44 45 46; do
  run_training s2_world_frame_termination "${seed}" \
    env.make_kwargs.check_healthy_z_relative=false
done

# C: S2 without the two explicit stair-climbing shaping terms.
for seed in 42 43 44 45 46; do
  run_training s2_no_shaping "${seed}" \
    env.make_kwargs.height_reward_weight=0 \
    env.make_kwargs.step_bonus=0
done
