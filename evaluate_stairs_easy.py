#!/usr/bin/env python3
"""
Evaluation script for stairs_easy configuration with deterministic policy.
This ensures consistent results across machines.
"""

import subprocess
import json
import sys

# Stairs easy configuration parameters (matching training config)
env_kwargs = {
    "flat_distance_before_stairs": 2.0,
    "num_steps": 8,
    "step_height": 0.10,
    "step_depth": 0.8,
    "end_platform_length": 5.0,
    "stairs_after_top": False,
    "end_with_abyss": False,
    "forward_reward_weight": 2.0,
    "height_reward_weight": 8.0,
    "ctrl_cost_weight": 0.1,
    "contact_cost_weight": 2e-7,
    "healthy_reward": 15.0,
    "step_bonus": 5.0,
    "lateral_penalty_weight": 0.3,
    "terminate_when_unhealthy": True,
    "healthy_z_range": [1.0, 2.0],
}

# Build the command with deterministic flag
cmd = [
    "python",
    "scripts/evaluate/evaluate_sb3.py",
    "--env_id", "HumanoidStairsConfigurable-v0",
    "--model_path", "outputs/2025-11-30/11-56-10/eval/best_model.zip",  # Best model (20M steps, reward: 12250.99)
    "--vecnorm_path", "outputs/2025-11-30/11-56-10/checkpoints/vecnormalize_11500000.pkl",  # From final checkpoint
    "--deterministic",  # Use deterministic policy for consistent results
    "--render",  # Show visualization
    "--episodes", "5",
    "--env_kwargs", json.dumps(env_kwargs),
]

print(f"Running evaluation with stairs_easy parameters (deterministic mode)...")
print(f"Environment kwargs: {json.dumps(env_kwargs, indent=2)}\n")

# Run the evaluation
result = subprocess.run(cmd)
sys.exit(result.returncode)
