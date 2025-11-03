# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a reinforcement learning project for training humanoid agents using MuJoCo physics simulation. The codebase supports two RL frameworks:

1. **Stable-Baselines3 (SB3)** - Primary framework with Hydra configuration management
2. **TorchRL** - Alternative implementation with native PyTorch

Both frameworks train PPO (Proximal Policy Optimization) agents on Gymnasium MuJoCo environments (Humanoid-v5, Walker2d-v5).

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- gymnasium[mujoco] >= 0.29
- mujoco >= 3.0.0
- stable-baselines3 >= 2.3.0
- hydra-core >= 1.3.2
- tensorboard >= 2.16

## Training Commands

### Stable-Baselines3 (SB3) Training

The primary training script (`train.py`) uses Hydra for configuration management. Config files are in `conf/`:

```bash
# Basic training (uses defaults from conf/main.yaml)
python train.py

# Train on Walker2d environment
python train.py env=walker2d \
  training.total_timesteps=5_000_000 \
  env.vec_env.n_envs=16 \
  algo.hyperparams.n_steps=1024 \
  algo.hyperparams.learning_rate=2.5e-4 \
  algo.hyperparams.ent_coef=0.005

# Train on Humanoid with GPU and extended timesteps
python train.py \
  env=humanoid \
  algo.device=cuda \
  training.total_timesteps=30_000_000 \
  env.vec_env.n_envs=16 \
  algo.hyperparams.n_steps=1024 \
  algo.hyperparams.learning_rate=2.5e-4 \
  algo.hyperparams.ent_coef=0.005
```

**Output Structure:**
- Training outputs are saved under `outputs/` with timestamped subdirectories
- Each run contains:
  - `final_model.zip` - Final trained model
  - `vecnormalize_final.pkl` - Normalization statistics
  - `checkpoints/` - Periodic checkpoints (`model_<steps>.zip`, `vecnormalize_<steps>.pkl`)
  - `eval/` - Evaluation logs and best model
  - TensorBoard logs

### TorchRL Training

Alternative training using native PyTorch implementation:

```bash
python torchrl_train_ppo.py \
  --env_id Humanoid-v5 \
  --total_frames 5_000_000 \
  --frames_per_batch 8192 \
  --minibatch 4096 \
  --n_envs 8 \
  --epochs 10 \
  --lr 2.5e-4 \
  --ent_coef 0.01 \
  --device cuda
```

**Output Structure:**
- Outputs saved to `outputs_torchrl/<env_id>/<timestamp>/`
- Contains: `policy_final.pt`, `value_final.pt`, `checkpoint_final.pt`
- Periodic checkpoints in `checkpoints/` subdirectory

## Evaluation Commands

### SB3 Evaluation

```bash
# Evaluate latest checkpoint (recommended)
./run_2d.sh

# Evaluate final model
./run_2d.sh --use-final-model

# Evaluate on specific environment
./run_2d.sh --env-id Humanoid-v5 --use-final-model

# Deterministic evaluation of best model
./run_2d.sh --use-final-model --env-id Humanoid-v5

# No rendering, non-deterministic
./run_2d.sh --no-render --no-deterministic

# Force specific run directory
./run_2d.sh --run outputs/2025-10-28/11-33-29
```

**Windows PowerShell:**
```powershell
.\run_2d.ps1                              # Latest checkpoint
.\run_2d.ps1 -UseFinalModel              # Final model
.\run_2d.ps1 -Best -EnvId Humanoid-v5    # Best model
.\run_2d.ps1 -Best -EnvId Humanoid-v5 -Deterministic  # Deterministic
```

### TorchRL Evaluation

```bash
python evaluate_torchrl.py \
  --env_id Humanoid-v5 \
  --checkpoint_path outputs_torchrl/.../checkpoint_final.pt \
  --episodes 5 \
  --render
```

## Monitoring Training

```bash
# Monitor SB3 training
tensorboard --logdir outputs

# Monitor TorchRL training
tensorboard --logdir outputs_torchrl

# Access TensorBoard
# Browser: http://localhost:6006/#timeseries
```

## Architecture Overview

### Stable-Baselines3 Pipeline

**Configuration System (Hydra):**
- `conf/main.yaml` - Main config with defaults (seed, vecnorm settings)
- `conf/env/` - Environment configs (humanoid.yaml, walker2d.yaml)
- `conf/algo/ppo.yaml` - PPO hyperparameters
- `conf/training/` - Training duration configs (default.yaml, long.yaml)

**Training Flow (train.py:22-149):**
1. Load Hydra config from `conf/`
2. Create vectorized environments with optional VecNormalize wrapper
3. Create evaluation environment with EvalCallback (saves best model)
4. Instantiate PPO model with hyperparameters from config
5. Train with CheckpointAndVecNormCallback (periodic saves)
6. Save final model and VecNormalize stats

**Environment Creation (utils/make_env.py):**
- `make_single_env()` - Creates single env with optional Monitor wrapper
- `make_vector_env()` - Creates SubprocVecEnv or DummyVecEnv
- **Custom Reward Wrapper:** `UprightAndEffortWrapper` applied to all envs (utils/make_env.py:16)
  - Adds upright bonus (weight=0.05) and effort penalty (weight=0.001)

**Callbacks (utils/callbacks.py):**
- `CheckpointAndVecNormCallback` - Saves model + VecNormalize every N steps
- Saves to `checkpoints/` as `model_<steps>.zip` and `vecnormalize_<steps>.pkl`

**VecNormalize Handling:**
- Normalizes observations and rewards during training
- Statistics saved alongside model checkpoints
- Must be loaded during evaluation for correct normalization
- Evaluation env uses `training=False` and `norm_reward=False`

### TorchRL Pipeline

**Training Flow (torchrl_train_ppo.py:102-327):**
1. Create parallel environments with DoubleToFloat transform
2. Build policy (actor) with Normal distribution sampling
3. Build critic (value function) with shared CastObs module
4. Collect rollouts with SyncDataCollector
5. Compute GAE advantages
6. Train with ClipPPOLoss using minibatch sampling
7. Save checkpoints periodically and log to TensorBoard

**Key Architecture Details:**
- **Dtype Casting:** CastObs module ensures float32 throughout (torchrl_train_ppo.py:44-47)
- **Policy:** MLP backbone → (loc, scale) → Independent Normal distribution
- **Critic:** Shared CastObs → ValueOperator
- **Gradient Clipping:** Applied to all parameters (default max_grad_norm=0.5)

## Key Implementation Details

### Reward Shaping

**UprightAndEffortWrapper (utils/reward_wrappers.py):**
```python
# Applied to ALL environments in make_single_env
env = UprightAndEffortWrapper(env, upright_w=0.05, effort_w=0.001)
```
This wrapper modifies the reward function to encourage upright posture and penalize excessive control effort.

### Batch Size Validation

Training enforces strict batch size divisibility (train.py:88-96):
```python
total_samples = n_steps * n_envs
assert total_samples % batch_size == 0
```
Ensure `batch_size` divides `n_steps * n_envs` when modifying configs.

### Evaluation Behavior

**SB3 Evaluation Scripts:**
- `run_2d.sh` / `run_2d.ps1` - Auto-detect latest SB3 run in `outputs/`
- Matches `vecnormalize_<steps>.pkl` with corresponding `model_<steps>.zip`
- Falls back to `final_model.zip` and `vecnormalize_final.pkl` if no checkpoints
- **Deterministic by default** for reproducible evaluation

**Direct Evaluation:**
```bash
python evaluate.py \
  --env_id Humanoid-v5 \
  --model_path outputs/.../model_1000000.zip \
  --vecnorm_path outputs/.../vecnormalize_1000000.pkl \
  --episodes 5 \
  --render \
  --deterministic
```

### VecNormalize Critical Notes

1. **Training and Evaluation Must Match:**
   - If training uses VecNormalize (enabled in conf/main.yaml), evaluation MUST load the corresponding .pkl file
   - Missing VecNormalize during evaluation causes incorrect observations → poor performance

2. **Evaluation Mode:**
   - Always set `training=False` and `norm_reward=False` for eval env (train.py:72-73)
   - Copy running stats from training env before evaluation

3. **Checkpoint Matching:**
   - Use `vecnormalize_<steps>.pkl` with `model_<steps>.zip`
   - Use `vecnormalize_final.pkl` with `final_model.zip`

## Common Tasks

### Adding a New Environment

1. Create config file: `conf/env/new_env.yaml`
2. Specify: `name`, `make_kwargs`, `vec_env` settings
3. Train: `python train.py env=new_env`

### Modifying PPO Hyperparameters

Edit `conf/algo/ppo.yaml` or override via command line:
```bash
python train.py \
  algo.hyperparams.learning_rate=1e-4 \
  algo.hyperparams.ent_coef=0.01
```

### Resuming Training

Not directly supported. To continue training:
1. Load the checkpoint in train.py with `PPO.load("path/to/model.zip", env=venv)`
2. Continue with `model.learn(...)` for additional timesteps

### Switching Between CPU/GPU

```bash
# SB3
python train.py algo.device=cuda

# TorchRL
python torchrl_train_ppo.py --device cuda
```
