# HumanoidStairs Environment Usage Guide

This document describes how to use the new HumanoidStairs environment for training agents to climb stairs.

## Overview

The HumanoidStairs environment adds a stair-climbing challenge to the standard Humanoid locomotion task. The environment features:

- **10-step staircase**: Each step is 0.6m deep and 0.15m high
- **Starting platform**: Flat area before stairs for the agent to begin
- **End platform**: Flat area after stairs for completing the climb
- **Custom reward shaping**: Rewards forward progress, height gained, and staying upright

## Training the Stairs Agent

### Basic Training Command

```bash
python train.py env=humanoid_stairs
```

### Training with Custom Parameters

```bash
python train.py env=humanoid_stairs \
  training.total_timesteps=10_000_000 \
  env.vec_env.n_envs=16 \
  algo.hyperparams.n_steps=1024 \
  algo.hyperparams.learning_rate=2.5e-4 \
  algo.hyperparams.ent_coef=0.005
```

### Training Output

Training runs will save outputs to `outputs/` with timestamped directories containing:
- `final_model.zip` - Final trained model
- `vecnormalize_final.pkl` - Normalization statistics
- `checkpoints/` - Periodic checkpoints during training
- TensorBoard logs

The training script now includes timing information that will display:
- Total training time
- Timesteps per second
- Total timesteps completed

## Training the Standard Humanoid (Walking)

The original Humanoid-v5 walking environment remains fully functional:

```bash
python train.py env=humanoid
```

## Environment Comparison

| Environment | Task | Observation Space | Action Space |
|-------------|------|-------------------|--------------|
| Humanoid-v5 | Walk on flat ground | 376 | 17 |
| HumanoidStairs-v0 | Climb stairs | 376 | 17 |

Both environments use the same observation and action spaces, making it easy to compare learning across tasks.

## Reward Structure (HumanoidStairs)

The stairs environment uses a composite reward function:

1. **Forward Progress** (weight: 1.25): Rewards positive x-velocity
2. **Height Gained** (weight: 2.0): Rewards increases in z-position
3. **Healthy Reward** (weight: 5.0): Bonus for staying upright
4. **Control Cost** (weight: 0.1): Penalty for excessive control effort

## Evaluation

### Auto-Evaluation During Training

The training script automatically evaluates the agent during training using the EvalCallback, saving the best model to the `eval/` subdirectory.

### Batch Evaluation

Evaluate all trained models (both Humanoid-v5 and HumanoidStairs-v0):

```bash
python evaluate_all_models.py
```

This script will:
- Find all trained models in the `outputs/` directory
- Auto-detect which environment each model was trained on
- Evaluate each model for 20 episodes
- Generate a comprehensive results report in `results.txt`

### Single Model Evaluation

```bash
python evaluate_stats.py \
  --env_id HumanoidStairs-v0 \
  --model_path outputs/2025-11-05/12-34-56/final_model.zip \
  --vecnorm_path outputs/2025-11-05/12-34-56/vecnormalize_final.pkl \
  --episodes 20 \
  --deterministic
```

## Testing the Environment

Test that the HumanoidStairs environment is correctly installed:

```bash
python test_stairs_env.py
```

This will verify:
- Environment registration
- Observation/action space dimensions
- Basic stepping functionality
- Info dictionary contents

## Files Added

The following new files were added for the stairs functionality:

```
envs/
├── __init__.py                    # Environment registration
├── humanoid_stairs.py             # HumanoidStairs environment class
└── assets/
    └── humanoid_stairs.xml        # MuJoCo physics model with stairs

conf/env/
└── humanoid_stairs.yaml           # Configuration for stairs training

test_stairs_env.py                 # Environment test script
STAIRS_USAGE.md                    # This file
```

## Configuration Files

### Existing Environments (Unchanged)

- `conf/env/humanoid.yaml` - Humanoid-v5 walking
- `conf/env/walker2d.yaml` - Walker2d-v5

### New Environment

- `conf/env/humanoid_stairs.yaml` - HumanoidStairs-v0 stair climbing

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir outputs
```

Then open http://localhost:6006 in your browser.

## Tips for Training

1. **Longer training**: Stair climbing is harder than walking. Consider training for 10M+ timesteps.
2. **Parallel environments**: Use more parallel environments (16-32) to speed up training.
3. **Hyperparameters**: The entropy coefficient (`ent_coef`) can help with exploration.
4. **Checkpoints**: Models are saved every 250k steps by default. You can evaluate intermediate checkpoints to track progress.

## Troubleshooting

### Import Error for `envs` module

Make sure you're running from the project root directory and the virtual environment is activated:

```bash
cd /path/to/rl-humanoid
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### MuJoCo rendering issues

If you encounter rendering issues, set `render_mode: null` in the config file or run training without rendering (default behavior).

### Environment not found

Ensure that the `envs` package is importable. The `train.py` script should automatically import it, but you can verify with:

```python
import envs
import gymnasium as gym
env = gym.make("HumanoidStairs-v0")
```
