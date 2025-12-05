#!/bin/bash
# Train the Humanoid agent with the solved configuration

echo "Starting training with fixed Z-range logic and improved rewards..."

python scripts/train/train_sb3.py \
    env=humanoid_stairs_solved \
    training.total_timesteps=20000000 \
    algo.hyperparams.ent_coef=0.01

echo "Training started. Check outputs/ for logs."

