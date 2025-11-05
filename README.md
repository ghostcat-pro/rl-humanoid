# RL Humanoid — SB3 + Hydra Skeleton

A minimal, reproducible setup for Gymnasium + Stable-Baselines3 with Hydra configuration.

Supports two environments:
- **Humanoid-v5**: Standard flat ground locomotion
- **HumanoidStairs-v0**: Custom stair climbing challenge

## Install
```bash
python -m venv .venv && source .venv/bin/activate # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Training

### Train on Flat Ground (Humanoid-v5)
```bash
python train.py env=humanoid \
  training.total_timesteps=10_000_000 \
  env.vec_env.n_envs=16 \
  algo.hyperparams.n_steps=1024 \
  algo.hyperparams.learning_rate=2.5e-4 \
  algo.hyperparams.ent_coef=0.005
```

### Train on Stairs (HumanoidStairs-v0)
```bash
python train.py env=humanoid_stairs \
  training.total_timesteps=10_000_000 \
  env.vec_env.n_envs=16 \
  algo.hyperparams.n_steps=1024 \
  algo.hyperparams.learning_rate=2.5e-4 \
  algo.hyperparams.ent_coef=0.005
```

### Train on Walker2d
```bash
python train.py env=walker2d \
  training.total_timesteps=5_000_000 \
  env.vec_env.n_envs=16
```

## Evaluation

### Evaluate Stairs Agent
```bash
# Automatically finds latest stairs-trained model
python evaluate_stairs.py

# Or specify exact run directory
python evaluate_stairs.py --run-dir outputs/2025-11-03/04-51-44

# More options
python evaluate_stairs.py --episodes 10 --no-render
```

### Evaluate Humanoid-v5 Agent
```bash
# Use evaluation script
python evaluate.py \
  --env_id Humanoid-v5 \
  --model_path outputs/2025-11-03/05-12-53/checkpoints/model_500000 \
  --vecnorm_path outputs/2025-11-03/05-12-53/checkpoints/vecnormalize_500000.pkl \
  --episodes 5 --render --deterministic
```

## Monitoring Training

```bash
# View training progress in TensorBoard
tensorboard --logdir outputs

# Access at http://localhost:6006/#timeseries
```

## Quick Testing

```bash
# Test stairs environment
python test_stairs_env.py

# Visual test (renders environment)
python test_stairs_visual.py