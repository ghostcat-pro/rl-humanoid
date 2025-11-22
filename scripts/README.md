# 🚀 Scripts Directory

Organized training, evaluation, and utility scripts for the RL Humanoid project.

---

## 📂 Directory Structure

```
scripts/
├── train/              # Training scripts
├── evaluate/           # Evaluation and testing scripts  
└── utils/              # PowerShell utilities and helpers
```

---

## 🏋️ Training Scripts (`train/`)

### `train_sb3.py`
Train humanoid agents using **Stable-Baselines3 (PPO)**.

**Environments:**
- Walker2d-v5 (2D walker)
- Humanoid-v5 (3D humanoid)
- HumanoidStairs-v0 (custom stairs)

**Usage:**
```bash
# Train Walker2d
python scripts/train/train_sb3.py env=walker2d

# Train Humanoid on stairs
python scripts/train/train_sb3.py env=humanoid training.total_timesteps=10000000

# Custom configuration
python scripts/train/train_sb3.py \
  env=walker2d \
  training.total_timesteps=2000000 \
  env.vec_env.n_envs=16 \
  algo.hyperparams.learning_rate=2.5e-4
```

---

### `train_destination.py`
Train the **HumanoidDestination-v0** goal navigation task.

**Usage:**
```bash
# Default (1M timesteps)
python scripts/train/train_destination.py

# Custom timesteps
python scripts/train/train_destination.py training.total_timesteps=5000000

# More parallel environments
python scripts/train/train_destination.py env.vec_env.n_envs=16
```

---

### `train_torchrl.py`
Train using **TorchRL** (alternative to SB3).

**Usage:**
```bash
# Train Walker2d
python scripts/train/train_torchrl.py --env_id Walker2d-v5 --total_frames 1000000

# Train with GPU
python scripts/train/train_torchrl.py \
  --env_id Humanoid-v5 \
  --total_frames 5000000 \
  --device cuda \
  --n_envs 16
```

---

## 🎯 Evaluation Scripts (`evaluate/`)

### `evaluate_sb3.py`
Evaluate Stable-Baselines3 trained models.

**Usage:**
```bash
# Evaluate specific model
python scripts/evaluate/evaluate_sb3.py \
  --env_id Walker2d-v5 \
  --model_path outputs/2025-10-27/14-03-03/final_model.zip \
  --vecnorm_path outputs/2025-10-27/14-03-03/vecnormalize_final.pkl \
  --episodes 10

# With rendering
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidDestination-v0 \
  --model_path outputs/best_model.zip \
  --render
```

---

### `evaluate_torchrl.py`
Evaluate TorchRL trained models.

**Usage:**
```bash
# Auto-find latest checkpoint
python scripts/evaluate/evaluate_torchrl.py --env_id Walker2d-v5 --n_episodes 5

# Specific checkpoint
python scripts/evaluate/evaluate_torchrl.py \
  --checkpoint outputs_torchrl/Walker2d-v5/2025-10-27/20-17-00/checkpoint_final.pt \
  --n_episodes 10 \
  --deterministic

# No rendering (faster)
python scripts/evaluate/evaluate_torchrl.py \
  --env_id Humanoid-v5 \
  --no_render \
  --n_episodes 20
```

---

### `evaluate_all.py`
Compare all trained models across environments.

**Usage:**
```bash
python scripts/evaluate/evaluate_all.py
```

---

### `evaluate_stats.py`
Generate statistics and performance metrics.

**Usage:**
```bash
python scripts/evaluate/evaluate_stats.py --run_dir outputs/2025-10-27/14-03-03
```

---

### `evaluate_video.py`
Create videos of trained agents.

**Usage:**
```bash
python scripts/evaluate/evaluate_video.py \
  --env_id HumanoidDestination-v0 \
  --model_path outputs/final_model.zip \
  --vecnorm_path outputs/vecnormalize_final.pkl \
  --video_dir videos/ \
  --episodes 3
```

---

## 🛠️ Utility Scripts (`utils/`)

### PowerShell Scripts

#### `run_sb3.ps1` (formerly `run_2dn.ps1`)
Quick evaluation of SB3 models.

**Usage:**
```powershell
# Latest model
.\scripts\utils\run_sb3.ps1

# Best model
.\scripts\utils\run_sb3.ps1 -Best

# Specific run
.\scripts\utils\run_sb3.ps1 -RunDir "2025-10-28/13-02-09"

# Different environment
.\scripts\utils\run_sb3.ps1 -EnvId HumanoidDestination-v0 -Best
```

---

#### `run_torchrl.ps1`
Quick evaluation of TorchRL models.

**Usage:**
```powershell
# Latest checkpoint
.\scripts\utils\run_torchrl.ps1 -EnvId Walker2d-v5

# Specific batch checkpoint
.\scripts\utils\run_torchrl.ps1 -Batch 50

# Custom settings
.\scripts\utils\run_torchrl.ps1 \
  -EnvId Humanoid-v5 \
  -Episodes 10 \
  -NoRender
```

---

### `create_video_gallery.py`
Create HTML gallery of training videos.

**Usage:**
```bash
python scripts/utils/create_video_gallery.py --video_dir videos/
```

---

## 🎓 Quick Start Examples

### 1. Train a new Walker2d agent
```bash
python scripts/train/train_sb3.py env=walker2d training.total_timesteps=1000000
```

### 2. Evaluate the trained model
```powershell
.\scripts\utils\run_sb3.ps1 -EnvId Walker2d-v5 -Best
```

### 3. Create a video
```bash
python scripts/evaluate/evaluate_video.py \
  --env_id Walker2d-v5 \
  --model_path outputs/2025-10-27/latest/best_model.zip \
  --vecnorm_path outputs/2025-10-27/latest/vecnormalize_final.pkl \
  --video_dir videos/
```

---

## 📝 Notes

- All training scripts save to timestamped directories in `outputs/` or `outputs_torchrl/`
- Evaluation scripts automatically find the latest checkpoints if not specified
- PowerShell scripts provide convenient wrappers for common evaluation tasks
- Use `--help` on any Python script to see all available options

---

## 🔗 See Also

- [Environments README](../envs/README.md) - Overview of all environments
- [Documentation](../docs/) - Detailed guides and analysis
- [Main README](../README.md) - Project overview
