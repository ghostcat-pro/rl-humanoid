# 🤖 RL Humanoid Locomotion

Reinforcement learning training for humanoid agents with **4 different locomotion tasks** of increasing complexity.

**Frameworks:** Stable-Baselines3 (SB3) + TorchRL  
**Configuration:** Hydra  
**Environments:** Custom Gymnasium/MuJoCo environments

---

## 🎯 Environments

This project includes **7 locomotion environments** with progressive difficulty:

### Base Environments

| Environment | Type | Difficulty | Observation Dims | Task |
|-------------|------|------------|------------------|------|
| **Walker2d-v5** | Built-in | ⭐ Easy | 17 | 2D bipedal walking |
| **Humanoid-v5** | Built-in | ⭐⭐ Medium | 376 | 3D forward walking |

### Challenge Environments

| Environment | Difficulty | Observation Dims | Task |
|-------------|------------|------------------|------|
| **HumanoidDestination-v0** | ⭐⭐⭐ Hard | 378 | Navigate to target (5m, 0m) |
| **HumanoidStairs-v0** | ⭐⭐⭐ Hard | 401 | Climb fixed 10-step staircase with 5×5 height grid |
| **HumanoidStairsConfigurable-v0** | ⭐⭐⭐⭐ Very Hard | 401 | Configurable stairs (height, depth, count, abyss) |
| **HumanoidCircuit-v0** | ⭐⭐⭐⭐⭐ Expert | 404 | Navigate waypoints + climb multiple staircases |

### Pre-configured Variants

**Stairs Configurations:**
- `humanoid_stairs_easy` - Lower, longer steps (8 steps, 10cm height)
- `humanoid_stairs_hard` - Steeper stairs (12 steps, 18cm height)
- `humanoid_stairs_short` - 5m approach (quick iterations)
- `humanoid_stairs_tiny` - 20 small steps (7.5cm each)
- `humanoid_stairs_abyss` - Must stop at top (no platform)
- `humanoid_stairs_updown` - Climb then descend

**Circuit Configurations:**
- `humanoid_circuit_flat` - 3 waypoints, no stairs
- `humanoid_circuit_simple` - 4 waypoints, 2 stair sections
- `humanoid_circuit_easy` - 3 waypoints, 1 gentle stair
- `humanoid_circuit_complex` - 6 waypoints with turns, 3 varied stairs
- `humanoid_circuit_custom` - 5 waypoints with waypoint 3 at stair top

📖 **[See detailed environment documentation →](envs/README.md)**

---

## 📦 Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Train an Agent

**Basic Environments:**
```powershell
# Train Walker2d (baseline, 1M steps)
python scripts/train/train_sb3.py env=walker2d

# Train standard Humanoid (10M steps)
python scripts/train/train_sb3.py env=humanoid training.total_timesteps=10000000
```

**Destination Navigation:**
```powershell
# Navigate to target position (20M steps recommended)
python scripts/train/train_destination.py training.total_timesteps=20000000
```

**Stairs Climbing:**
```powershell
# Original fixed stairs
python scripts/train/train_sb3.py env=humanoid_stairs training.total_timesteps=10000000

# Easy stairs (lower steps, good for learning)
python scripts/train/train_sb3.py env=humanoid_stairs_easy training.total_timesteps=5000000

# Hard stairs (steeper, more challenging)
python scripts/train/train_sb3.py env=humanoid_stairs_hard training.total_timesteps=15000000

# Custom configuration
python scripts/train/train_sb3.py env=humanoid_stairs_configurable training.total_timesteps=10000000
```

**Circuit Navigation:**
```powershell
# Simple circuit (4 waypoints, 2 stairs)
python scripts/train/train_sb3.py env=humanoid_circuit_simple training.total_timesteps=20000000

# Complex circuit (6 waypoints with turns, 3 stairs)
python scripts/train/train_sb3.py env=humanoid_circuit_complex training.total_timesteps=30000000

# Custom circuit (5 waypoints, waypoint 3 on stair top)
python scripts/train/train_sb3.py env=humanoid_circuit_custom training.total_timesteps=25000000
```

**Resume Training:**
```powershell
# Resume from checkpoint (calculates remaining steps automatically)
python scripts/train/train_sb3.py env=humanoid_stairs_easy \
  resume_from="outputs/2025-11-22/00-57-36/checkpoints/model_750000.zip" \
  training.total_timesteps=5000000

# Resume destination training
python scripts/train/train_destination.py \
  resume_from="outputs/2025-11-21/09-38-36/checkpoints/model_13250000.zip" \
  training.total_timesteps=20000000
```

### 2. Visualize Trained Agent

**Live Rendering:**
```powershell
# View latest stairs training
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidStairsConfigurable-v0 \
  --model_path "outputs/2025-11-22/00-57-36/checkpoints/model_750000.zip" \
  --vecnorm_path "outputs/2025-11-22/00-57-36/checkpoints/vecnormalize_750000.pkl" \
  --render --deterministic --episodes 5

# View destination navigation
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidDestination-v0 \
  --model_path "outputs/2025-11-21/21-03-31/checkpoints/model_28500000.zip" \
  --vecnorm_path "outputs/2025-11-21/21-03-31/checkpoints/vecnormalize_28500000.pkl" \
  --render --deterministic --episodes 3

# View circuit navigation
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidCircuit-v0 \
  --model_path "path/to/checkpoint.zip" \
  --vecnorm_path "path/to/vecnormalize.pkl" \
  --render --episodes 3
```

**Record Videos:**
```powershell
# Create video recordings
python scripts/evaluate/evaluate_video.py \
  --env_id HumanoidStairsConfigurable-v0 \
  --model_path "outputs/2025-11-22/00-57-36/checkpoints/model_750000.zip" \
  --vecnorm_path "outputs/2025-11-22/00-57-36/checkpoints/vecnormalize_750000.pkl" \
  --video_dir "./videos/stairs" \
  --episodes 3 \
  --deterministic
```

### 3. Monitor Training

**TensorBoard:**
```powershell
# View all training runs
tensorboard --logdir outputs

# View specific date
tensorboard --logdir outputs/2025-11-22

# View specific run
tensorboard --logdir outputs/2025-11-22/00-57-36
```
Open http://localhost:6006

---

## 📂 Project Structure

```
rl-humanoid/
├── envs/                        # Custom environments
│   ├── README.md               # Environment documentation
│   ├── custom/                 # Custom environment implementations
│   │   ├── humanoid_stairs.py
│   │   └── humanoid_destination.py
│   └── assets/                 # MuJoCo XML models
│       ├── humanoid_stairs.xml
│       └── humanoid_destination.xml
├── scripts/                     # Training & evaluation scripts
│   ├── README.md               # Scripts documentation
│   ├── train/                  # Training scripts
│   │   ├── train_sb3.py       # Stable-Baselines3 trainer
│   │   ├── train_destination.py
│   │   └── train_torchrl.py   # TorchRL trainer
│   ├── evaluate/               # Evaluation scripts
│   │   ├── evaluate_sb3.py
│   │   ├── evaluate_torchrl.py
│   │   ├── evaluate_all.py
│   │   ├── evaluate_stats.py
│   │   └── evaluate_video.py
│   └── utils/                  # Utility scripts
│       ├── run_sb3.ps1        # PowerShell evaluation helpers
│       ├── run_torchrl.ps1
│       └── create_video_gallery.py
├── conf/                        # Hydra configuration
│   ├── main.yaml
│   ├── env/                    # Environment configs
│   ├── algo/                   # Algorithm configs (PPO)
│   └── training/               # Training configs
├── utils/                       # Python utilities
│   ├── make_env.py
│   ├── callbacks.py
│   └── vecnorm_io.py
├── docs/                        # Documentation
│   ├── STAIRS_USAGE.md
│   ├── walking_humanoid.md
│   └── *.md
├── outputs/                     # SB3 training outputs
├── outputs_torchrl/            # TorchRL training outputs
├── videos/                      # Generated videos
├── train.py                     # Legacy training script (use scripts/train/ instead)
├── evaluate.py                  # Legacy eval script (use scripts/evaluate/ instead)
└── README.md                    # This file
```

---

## 📖 Documentation

- **[Environments](envs/README.md)** - Detailed environment documentation
- **[Scripts](scripts/README.md)** - Training & evaluation guide
- **[Configurable Stairs](docs/CONFIGURABLE_STAIRS.md)** - Full stairs parameterization guide
- **[Circuit Environment](docs/CIRCUIT_ENVIRONMENT.md)** - Waypoint navigation with obstacles
- **[Reward Functions Analysis](docs/REWARD_FUNCTIONS_ANALYSIS.md)** - Comparison across all environments
- **[Stairs Height Grid](docs/STAIRS_HEIGHT_GRID_VISUALIZATION.md)** - 5×5 terrain perception mechanics
- **[Stairs Usage](docs/STAIRS_USAGE.md)** - Original stairs environment guide
- **[Walking Humanoid](docs/walking_humanoid.md)** - Destination task walkthrough

---

## 🎓 Training Examples

### Walker2d (Baseline)
```powershell
python scripts/train/train_sb3.py env=walker2d training.total_timesteps=1000000
```

### Humanoid Forward Walking
```powershell
python scripts/train/train_sb3.py \
  env=humanoid \
  training.total_timesteps=10000000 \
  env.vec_env.n_envs=16
```

### Destination Navigation
```powershell
# Full 20M training
python scripts/train/train_destination.py training.total_timesteps=20000000

# Resume if interrupted
python scripts/train/train_destination.py \
  resume_from="outputs/2025-11-21/09-38-36/checkpoints/model_13250000.zip" \
  training.total_timesteps=20000000
```

### Stairs Climbing

**Easy Stairs (Learning):**
```powershell
python scripts/train/train_sb3.py \
  env=humanoid_stairs_easy \
  training.total_timesteps=5000000 \
  env.vec_env.n_envs=8
```

**Standard Stairs:**
```powershell
python scripts/train/train_sb3.py \
  env=humanoid_stairs \
  training.total_timesteps=10000000 \
  env.vec_env.n_envs=16
```

**Hard Stairs (Challenge):**
```powershell
python scripts/train/train_sb3.py \
  env=humanoid_stairs_hard \
  training.total_timesteps=15000000 \
  env.vec_env.n_envs=16
```

### Circuit Navigation

**Simple Circuit:**
```powershell
python scripts/train/train_sb3.py \
  env=humanoid_circuit_simple \
  training.total_timesteps=20000000 \
  env.vec_env.n_envs=16
```

**Custom Circuit (5 waypoints with stairs):**
```powershell
python scripts/train/train_sb3.py \
  env=humanoid_circuit_custom \
  training.total_timesteps=25000000 \
  env.vec_env.n_envs=16
```

**Complex Circuit (Expert):**
```powershell
python scripts/train/train_sb3.py \
  env=humanoid_circuit_complex \
  training.total_timesteps=30000000 \
  env.vec_env.n_envs=16 \
  training=long
```

### TorchRL Training
```bash
python scripts/train/train_torchrl.py \
  --env_id Walker2d-v5 \
  --total_frames 1000000 \
  --n_envs 16 \
  --device cpu
```

---

## 🔧 Configuration

Use Hydra to override any parameter:

```powershell
# Use longer training preset
python scripts/train/train_sb3.py env=humanoid training=long

# Custom hyperparameters
python scripts/train/train_sb3.py \
  env=walker2d \
  algo.hyperparams.learning_rate=2.5e-4 \
  algo.hyperparams.batch_size=4096

# More parallel environments for faster training
python scripts/train/train_sb3.py \
  env=humanoid \
  env.vec_env.n_envs=32

# Adjust checkpoint frequency
python scripts/train/train_sb3.py \
  env=humanoid_stairs \
  training.checkpoint_every_steps=500000
```

### Key Configuration Files

- `conf/env/*.yaml` - Environment configurations
  - Base: `humanoid.yaml`, `walker2d.yaml`
  - Stairs: `humanoid_stairs_*.yaml` (easy, hard, configurable, etc.)
  - Circuit: `humanoid_circuit_*.yaml` (flat, simple, complex, custom)
  - Destination: Configured in `train_destination.py`
- `conf/algo/ppo.yaml` - PPO algorithm hyperparameters
- `conf/training/*.yaml` - Training duration settings (default, long)
- `conf/main.yaml` - Global settings (seed, vecnorm, resume_from)

---

## 💡 Tips

1. **Start simple** - Train Walker2d-v5 first to verify setup
2. **Progressive difficulty** - Master easier stairs before attempting hard/circuit
3. **Use checkpoints** - Models auto-saved every 250k steps in `outputs/<date>/<time>/checkpoints/`
4. **Resume interrupted training** - Use `resume_from="path/to/checkpoint.zip"` parameter
5. **Monitor with TensorBoard** - Real-time training visualization with `tensorboard --logdir outputs`
6. **VecNormalize preservation** - Resume function now loads both model and normalization stats
7. **Observation spaces**:
   - Height grid (stairs/circuit): 5×5 grid, 0.3m spacing, 25 additional dims
   - Waypoint vector (circuit): 2D relative coordinates to current target
   - Target vector (destination): 2D relative coordinates to goal
8. **Training duration recommendations**:
   - Walker2d: 1M steps (~30 min)
   - Humanoid-v5: 5-10M steps (~5-10 hours)
   - Destination: 15-20M steps (~15-20 hours)
   - Stairs (easy): 5M steps (~5 hours)
   - Stairs (standard/hard): 10-15M steps (~10-15 hours)
   - Circuit (simple): 20M steps (~20 hours)
   - Circuit (complex): 30M+ steps (~30+ hours)
9. **Stairs configurations**: Start with `humanoid_stairs_easy` for learning, then progress to standard/hard
10. **Circuit difficulty**: Use curriculum from flat → simple → easy → complex

---

## 🐛 Troubleshooting

**Import errors:**
- Always activate virtual environment: `.\.venv\Scripts\activate`
- Make sure to run from project root

**MuJoCo rendering issues:**
```bash
# Linux/Mac
export MUJOCO_GL=glfw

# Windows (usually works out of box)
```

**Environment not found:**
- Scripts in `scripts/` automatically import `envs` module
- Legacy scripts in root may need manual import

---

## 📊 Performance Benchmarks

| Environment | Timesteps | Mean Reward | Observation Dims | Training Time* |
|-------------|-----------|-------------|------------------|----------------|
| Walker2d-v5 | 1M | ~4500 | 17 | ~30 min |
| Humanoid-v5 | 5M | ~5000 | 376 | ~5 hours |
| HumanoidDestination-v0 | 20M | ~8000+ | 378 (376 + 2 target) | ~20 hours |
| HumanoidStairs-v0 | 10M | ~6000+ | 401 (376 + 25 grid) | ~10 hours |
| HumanoidStairsEasy | 5M | ~5000+ | 401 | ~5 hours |
| HumanoidStairsHard | 15M | ~7000+ | 401 | ~15 hours |
| HumanoidCircuitSimple | 20M | ~4000+ | 404 (376 + 25 grid + 2 waypoint + 1 progress) | ~20 hours |
| HumanoidCircuitComplex | 30M+ | ~3000+ | 404 | ~30+ hours |

*Approximate on 8-16 parallel environments with CPU

### Key Observations

- **Height grid** adds 25 observations (5×5 grid, 0.3m spacing around agent)
- **Destination task** uses relative coordinates for generalization
- **Circuit tasks** combine navigation + stair climbing skills
- **Contact cost** affects movement quality (configured per environment)
- **Resume training** preserves both model weights and VecNormalize statistics

---

## 🤝 Contributing

This is a research project. Feel free to:
- Add new environments
- Improve reward shaping
- Add new algorithms
- Enhance documentation

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Gymnasium** - https://gymnasium.farama.org/
- **MuJoCo** - https://mujoco.org/
- **Stable-Baselines3** - https://stable-baselines3.readthedocs.io/
- **TorchRL** - https://pytorch.org/rl/
- **Hydra** - https://hydra.cc/