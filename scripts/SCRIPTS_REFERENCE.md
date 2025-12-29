# Scripts Reference Guide

Complete functionality reference for all scripts in the `scripts/` directory.

---

## 📁 Directory Structure

```
scripts/
├── train/           # Training scripts
├── evaluate/        # Evaluation scripts
└── utils/          # Utility scripts and tools
```

---

## 🎓 Training Scripts (`scripts/train/`)

### `train_sb3.py`
**Purpose:** Main training script using Stable-Baselines3 (PPO algorithm)

**Functionality:**
- Trains agents on any environment using Hydra configuration
- Supports all custom environments (stairs, circuit, destination)
- Automatic checkpointing every 250k steps
- VecNormalize state saving
- TensorBoard logging
- Resume training from checkpoints

**Usage:**
```bash
python scripts/train/train_sb3.py env=humanoid training.total_timesteps=50000000
```

**Key Features:**
- Hydra config management
- Multi-environment vectorization
- Periodic evaluation during training
- Saves best model based on eval performance

---

### `train_destination.py`
**Purpose:** Specialized training for HumanoidDestination-v0 environment

**Functionality:**
- Trains agent to navigate to target position (10m, 0m)
- Uses DestinationWrapper for episode termination on success
- Same PPO training pipeline as train_sb3.py
- Optimized for navigation tasks

**Usage:**
```bash
python scripts/train/train_destination.py training.total_timesteps=20000000
```

**Key Features:**
- Automatic destination wrapper integration
- Success-based episode termination
- Saves vecnormalize state for evaluation

---

### `train_torchrl.py`
**Purpose:** Experimental training script using TorchRL framework

**Functionality:**
- PPO implementation using TorchRL primitives
- Alternative to Stable-Baselines3
- Used for initial experiments only
- Supports Walker2d-v5 and Humanoid-v5

**Usage:**
```bash
python scripts/train/train_torchrl.py --env Walker2d-v5 --total-steps 1000000
```

**Status:** Experimental - main project uses Stable-Baselines3

---

## 📊 Evaluation Scripts (`scripts/evaluate/`)

### `evaluate_sb3.py`
**Purpose:** Evaluate trained models with live rendering or headless mode

**Functionality:**
- Load trained PPO models and VecNormalize state
- Run episodes with deterministic or stochastic policy
- Display live rendering or run headless
- Custom environment parameters via `--env_kwargs` JSON
- Print detailed episode statistics

**Usage:**
```bash
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidCircuit-v0 \
  --model_path outputs/2025-12-23/14-37-50/eval/best_model.zip \
  --vecnorm_path outputs/2025-12-23/14-37-50/vecnormalize_final.pkl \
  --render --deterministic --episodes 5
```

**Key Features:**
- Supports all environments (Humanoid-v5, stairs, circuit, destination)
- Custom environment kwargs for configurable environments
- Deterministic evaluation for reproducibility
- Episode reward and length statistics

---

### `evaluate_stats.py`
**Purpose:** Statistical evaluation without rendering (headless mode)

**Functionality:**
- Run multiple episodes and collect statistics
- Calculate mean/std reward and episode length
- No rendering (faster evaluation)
- Verbose mode for step-by-step debugging

**Usage:**
```bash
python scripts/evaluate/evaluate_stats.py \
  --env_id Humanoid-v5 \
  --model_path outputs/.../best_model.zip \
  --episodes 20 --deterministic
```

**Output:**
- Mean reward ± std dev
- Mean episode length
- Per-episode breakdown (with --verbose)

---

### `evaluate_video.py`
**Purpose:** Evaluate models and record MP4 videos

**Functionality:**
- Same as evaluate_sb3.py but records videos
- Creates MP4 files for each episode
- Useful for creating demonstrations
- Custom video directory

**Usage:**
```bash
python scripts/evaluate/evaluate_video.py \
  --env_id HumanoidStairsConfigurable-v0 \
  --model_path outputs/.../best_model.zip \
  --vecnorm_path outputs/.../vecnormalize_final.pkl \
  --video_dir ./videos/stairs \
  --episodes 3 --deterministic
```

**Output:** MP4 video files in specified directory

---

### `evaluate_torchrl.py`
**Purpose:** Evaluate TorchRL-trained models

**Functionality:**
- Evaluate models trained with train_torchrl.py
- TorchRL environment wrapper
- Live rendering

**Usage:**
```bash
python scripts/evaluate/evaluate_torchrl.py \
  --env Walker2d-v5 \
  --model-path outputs_torchrl/.../actor_critic.pt \
  --episodes 5
```

**Status:** Experimental - matches train_torchrl.py

---

### `evaluate_all.py`
**Purpose:** Batch evaluation of all trained models in outputs directory

**Functionality:**
- Automatically discovers all final_model.zip files
- Evaluates each model (20 episodes)
- Generates comprehensive results report
- Creates summary table sorted by performance

**Usage:**
```bash
python scripts/evaluate/evaluate_all.py
```

**Output:**
- Results table with mean rewards
- Best model identification
- Saves report to text file

---

## 🛠️ Utility Scripts (`scripts/utils/`)

### `evaluate_circuit_flat.py`
**Purpose:** Quick evaluation shortcut for circuit_flat best model

**Functionality:**
- Pre-configured env_kwargs for circuit_flat
- Calls evaluate_sb3.py with correct parameters
- Hardcoded to best model path

**Usage:**
```bash
python scripts/utils/evaluate_circuit_flat.py
```

---

### `evaluate_stairs_easy.py`
**Purpose:** Quick evaluation shortcut for stairs_easy best model

**Functionality:**
- Pre-configured env_kwargs for stairs_easy (8 steps, 10cm height)
- Calls evaluate_sb3.py with correct parameters
- Hardcoded to best model path

**Usage:**
```bash
python scripts/utils/evaluate_stairs_easy.py
```

---

### `check_waypoints.py`
**Purpose:** Debug waypoint reaching behavior in circuit environment

**Functionality:**
- Detailed tracking of waypoint reaching events
- Reward breakdown by source (waypoints, circuit completion, per-step)
- Checks circuit completion status
- Analyzes specific circuit_flat model

**Usage:**
```bash
python scripts/utils/check_waypoints.py
```

**Output:**
- Waypoints reached count
- Circuit completion status
- Detailed reward breakdown

---

### `visualize_stairs_raw.py`
**Purpose:** Visualize stairs environment with random actions (no trained model)

**Functionality:**
- Creates stairs environment with specific configuration
- Runs random actions to show environment setup
- Useful for verifying stairs geometry
- Camera control testing

**Usage:**
```bash
python scripts/utils/visualize_stairs_raw.py
```

**Use Cases:**
- Environment setup verification
- Stairs geometry debugging
- Testing rendering

---

### `create_video_gallery.py`
**Purpose:** Create HTML gallery to view video frames in browser

**Functionality:**
- Generates interactive HTML page from extracted frames
- Workaround for OpenGL/display issues
- Frame-by-frame navigation
- Automatic slideshow mode

**Usage:**
```bash
python scripts/utils/create_video_gallery.py frames_output/
```

**Output:** gallery.html with embedded frame viewer

---

### `extract_frames.sh`
**Purpose:** Extract individual frames from MP4 videos using ffmpeg

**Functionality:**
- Extracts frames at specified frame rate (default 5 fps)
- Saves as PNG images
- Creates contact sheets/montages
- Linux/Mac bash script

**Usage:**
```bash
./scripts/utils/extract_frames.sh videos/eval-episode.mp4 [output_dir] [frame_rate]
```

**Requirements:** ffmpeg, imagemagick (for montage)

---

### `run_2d.ps1`
**Purpose:** PowerShell script for quick model evaluation (Windows)

**Functionality:**
- Windows PowerShell wrapper for evaluate_sb3.py
- Simplified parameter passing
- Supports best model and final model options

**Usage:**
```powershell
.\scripts\utils\run_2d.ps1 -Best -EnvId Humanoid-v5 -Deterministic
```

---

### `run_2dn.ps1`
**Purpose:** PowerShell script variant (Windows)

**Functionality:**
- Alternative PowerShell evaluation wrapper
- Similar to run_2d.ps1

---

### `tests/`
**Purpose:** Test files for environment functionality

**Contents:**
- test_circuit.py
- test_configurable_stairs.py
- test_destination_termination.py
- test_stairs_distance.py
- test_stairs_env.py
- test_stairs_solved.py

See [scripts/utils/tests/README.md](scripts/utils/tests/README.md) for details.

---

## 🔑 Quick Reference

### Common Tasks

**Train a model:**
```bash
python scripts/train/train_sb3.py env=humanoid_stairs_easy training.total_timesteps=30000000
```

**Evaluate with rendering:**
```bash
python scripts/evaluate/evaluate_sb3.py --env_id HumanoidStairsConfigurable-v0 \
  --model_path outputs/.../best_model.zip --vecnorm_path outputs/.../vecnormalize_final.pkl \
  --render --deterministic --episodes 5
```

**Record video:**
```bash
python scripts/evaluate/evaluate_video.py --env_id Humanoid-v5 \
  --model_path outputs/.../best_model.zip --video_dir ./videos --episodes 3
```

**Get statistics:**
```bash
python scripts/evaluate/evaluate_stats.py --env_id Humanoid-v5 \
  --model_path outputs/.../best_model.zip --episodes 20 --deterministic
```

**Batch evaluate all models:**
```bash
python scripts/evaluate/evaluate_all.py
```

---

## 📝 Notes

- **Main framework:** All primary training/evaluation uses Stable-Baselines3
- **TorchRL scripts:** Experimental only, not used for main results
- **Shortcuts:** evaluate_circuit_flat.py and evaluate_stairs_easy.py provide quick access to best models
- **Platform support:** Most scripts work on Windows/Linux/Mac; .ps1 scripts are Windows-only
- **Custom environments:** All scripts support custom environment kwargs via JSON
