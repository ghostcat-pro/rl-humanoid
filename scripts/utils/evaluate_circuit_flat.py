#!/usr/bin/env python3
"""
Evaluation script for circuit_flat environment.
"""

import subprocess
import json
import sys

# Circuit flat configuration parameters
env_kwargs = {
    "waypoints": [[10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
    "waypoint_reach_threshold": 1.0,
    "stairs": [],
    "terrain_width": 30.0,
    "progress_reward_weight": 200.0,
    "waypoint_bonus": 100.0,
    "height_reward_weight": 0.0,
    "forward_reward_weight": 0.5,
    "ctrl_cost_weight": 0.1,
    "contact_cost_weight": 5e-7,
    "healthy_reward": 5.0,
    "terminate_when_unhealthy": True,
    "healthy_z_range": [0.8, 3.0],
}

# Build the command with deterministic flag
cmd = [
    "python",
    "scripts/evaluate/evaluate_sb3.py",
    "--env_id", "HumanoidCircuit-v0",
    "--model_path", "outputs/2025-12-12/09-48-00/eval/best_model.zip",
    "--vecnorm_path", "outputs/2025-12-12/09-48-00/vecnormalize_final.pkl",
    "--deterministic",
    "--render",
    "--episodes", "5",
    "--env_kwargs", json.dumps(env_kwargs),
]

print(f"Running evaluation with circuit_flat parameters (deterministic mode)...")
print(f"Environment kwargs: {json.dumps(env_kwargs, indent=2)}\n")

# Run the evaluation
result = subprocess.run(cmd)
sys.exit(result.returncode)
