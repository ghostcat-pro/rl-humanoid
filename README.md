# 🤖 RL Humanoid Locomotion

Reinforcement learning training for humanoid agents with **4 different locomotion tasks** of increasing complexity.

**Primary Framework:** Stable-Baselines3 (SB3) with PPO
**Configuration:** Hydra
**Environments:** Custom Gymnasium/MuJoCo environments

> **Note:** TorchRL and SAC were used only for initial experiments. All main training and results use Stable-Baselines3 with PPO. See [Extras](#-extras-sac--torchrl) for running these alternative approaches.



---

## 🎯 Environments

This project includes **7 locomotion environments** with progressive difficulty:

### Base Environments

| Environment | Type | Difficulty | Observation Dims | Task |
|-------------|------|------------|------------------|------|
| **Walker2d-v5** | Built-in | ⭐ Easy | 17 | 2D bipedal walking |
| **Humanoid-v5** | Built-in | ⭐⭐ Medium | 376 | 3D forward walking |

### Challenge Environments

| Environment | Difficulty | Observation Dims | Task Description |
|-------------|------------|------------------|------------------|
| **HumanoidDestination-v0** | ⭐⭐⭐ Hard | **378** | Navigate to target (10m, 0m) - terminates on success |
| **HumanoidStairs-v0** | ⭐⭐⭐ Hard | **401** | Climb fixed 10-step staircase with 5×5 height grid |
| **HumanoidStairsConfigurable-v0** | ⭐⭐⭐⭐ Very Hard | **401** | Configurable stairs (height, depth, count, abyss) |
| **HumanoidCircuit-v0** | ⭐⭐⭐⭐⭐ Expert | **406** | Navigate waypoints + climb multiple staircases |

### Observation Space Details

- **Base Humanoid (376):** Position, velocity, COM inertia/velocity, actuator forces, contact forces
- **+2 (Destination):** Relative vector to target position (dx, dy)
- **+25 (Stairs/Circuit):** 5×5 height grid for terrain perception (0.3m spacing, agent-centered)
- **+5 (Circuit):** Waypoint vector (2), progress (1), heading error sin/cos (2)

📖 **[See detailed observation space documentation →](docs/OBSERVATION_SPACES.md)**

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

**Note on Commands:** Examples below show PowerShell (Windows) syntax. For Linux/Mac bash, replace line continuation `` ` `` with `\`.

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
# PowerShell (Windows)
python scripts/train/train_sb3.py env=humanoid_stairs_easy `
  resume_from="outputs/2025-11-22/00-57-36/checkpoints/model_750000.zip" `
  training.total_timesteps=5000000

# Bash (Linux/Mac)
python scripts/train/train_sb3.py env=humanoid_stairs_easy \
  resume_from="outputs/2025-11-22/00-57-36/checkpoints/model_750000.zip" \
  training.total_timesteps=5000000
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
# PowerShell (Windows)
tensorboard --logdir outputs
```

```bash
# Bash (Linux/Mac)
tensorboard --logdir outputs
```

Open http://localhost:6006 to view training metrics, loss curves, and episode rewards.

---

## 📂 Project Structure

```
rl-humanoid/
├── envs/                           # Custom environments
│   ├── README.md                  # Environment documentation
│   ├── custom/                    # Custom environment implementations
│   │   ├── humanoid_destination.py
│   │   ├── humanoid_stairs.py
│   │   ├── humanoid_stairs_configurable.py
│   │   └── humanoid_circuit.py
│   └── assets/                    # MuJoCo XML models
│       ├── humanoid_destination.xml
│       ├── humanoid_stairs.xml
│       └── humanoid_circuit.xml
├── scripts/                        # Training & evaluation scripts
│   ├── SCRIPTS_REFERENCE.md       # Complete scripts functionality guide
│   ├── README.md                  # Scripts documentation
│   ├── train/                     # Training scripts
│   │   ├── train_sb3.py          # Main SB3 trainer (Hydra config)
│   │   ├── train_destination.py  # Destination navigation trainer
│   │   ├── train_sac.py          # SAC trainer (experimental)
│   │   └── train_torchrl.py      # TorchRL trainer (experimental)
│   ├── evaluate/                  # Evaluation scripts
│   │   ├── evaluate_sb3.py       # Main evaluation with rendering
│   │   ├── evaluate_stats.py     # Statistical evaluation (headless)
│   │   ├── evaluate_video.py     # Record MP4 videos
│   │   ├── evaluate_sac.py       # SAC evaluation (experimental)
│   │   ├── evaluate_torchrl.py   # TorchRL evaluation (experimental)
│   │   └── evaluate_all.py       # Batch evaluate all models
│   └── utils/                     # Utility scripts
│       ├── evaluate_circuit_flat.py    # Quick circuit evaluation
│       ├── evaluate_stairs_easy.py     # Quick stairs evaluation
│       ├── check_waypoints.py          # Debug waypoint behavior
│       ├── visualize_stairs_raw.py     # Visualize env (no model)
│       ├── create_video_gallery.py     # HTML frame gallery
│       ├── extract_frames.sh           # Extract frames from videos
│       ├── run_2d.ps1                  # PowerShell evaluation wrapper
│       ├── run_2dn.ps1                 # PowerShell evaluation variant
│       └── tests/                      # Environment test files
│           └── README.md
├── conf/                           # Hydra configuration
│   ├── main.yaml                  # Main config entry point
│   ├── env/                       # Environment configs
│   │   ├── humanoid.yaml
│   │   ├── humanoid_sac.yaml     # SAC-specific config
│   │   ├── humanoid_stairs_*.yaml
│   │   └── humanoid_circuit_*.yaml
│   ├── algo/                      # Algorithm configs
│   │   ├── ppo.yaml              # PPO (main algorithm)
│   │   └── sac.yaml              # SAC (experimental)
│   └── training/                  # Training configs
│       └── *.yaml
├── utils/                          # Python utilities
│   ├── make_env.py                # Environment creation
│   ├── callbacks.py               # Training callbacks
│   ├── vecnorm_io.py              # VecNormalize serialization
│   ├── reward_wrappers.py         # Reward modification wrappers
│   └── destination_wrapper.py     # Destination termination wrapper
├── docs/                           # Documentation
│   ├── OBSERVATION_SPACES.md      # Complete obs space breakdown
│   ├── REWARD_FUNCTIONS_ANALYSIS.md  # Reward comparison
│   ├── TRAINING_LOG.md            # 58 training experiments history
│   ├── CONFIGURABLE_STAIRS.md     # Stairs parameterization
│   ├── CIRCUIT_ENVIRONMENT.md     # Circuit navigation guide
│   ├── STAIRS_HEIGHT_GRID_VISUALIZATION.md  # Height grid mechanics
│   ├── STAIRS_USAGE.md            # Original stairs guide
│   ├── walking_humanoid.md        # Destination walkthrough
│   └── archive/                   # Historical documents
│       ├── README.md              # Archive explanation
│       ├── stairs_training_nov23-24.md
│       ├── stairs_debugging_nov23.md
│       ├── stairs_health_check_fix.md
│       ├── training_summary_dec12.md
│       └── evaluation_results_humanoid_oct28.txt
├── examples/                       # Example scripts
├── outputs/                        # SB3 training outputs (local)
├── outputs_best/                   # Best models (tracked in git)
│   └── 2025-XX-XX/                # Best model checkpoints
├── outputs_destination/            # Destination training runs
├── outputs_sac/                    # SAC training outputs (experimental)
├── outputs_torchrl/                # TorchRL training outputs (experimental)
├── videos/                         # Generated videos
│   ├── circuit_flat_80m/
│   ├── stairs_easy_10m/
│   └── *.mp4
├── best_videos/                    # Best result videos (tracked in git)
├── how_to_use.txt                  # Best model commands reference
├── INSPIRATION.md                  # Project inspiration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

**Note on `outputs_best/`:**  
This folder contains only the **4 best models** for each scenario and is tracked in git for easy sharing. After cloning the repository, copy the contents to your local `outputs/` folder to use these pre-trained models:

```powershell
# PowerShell (Windows)
Copy-Item -Recurse outputs_best\* outputs\
```

```bash
# Bash (Linux/Mac)
cp -r outputs_best/* outputs/
```

## 📖 Documentation

- **[Scripts Reference](scripts/SCRIPTS_REFERENCE.md)** - Complete functionality guide for all training, evaluation, and utility scripts
- **[Environments](envs/README.md)** - Detailed environment documentation
- **[Observation Spaces](docs/OBSERVATION_SPACES.md)** - Complete observation space breakdown for all environments
- **[Reward Functions Analysis](docs/REWARD_FUNCTIONS_ANALYSIS.md)** - Detailed comparison of reward functions across all environments
- **[Training Log](docs/TRAINING_LOG.md)** - Comprehensive history of 58 training experiments (Oct-Dec 2025)
- **[Configurable Stairs](docs/CONFIGURABLE_STAIRS.md)** - Full stairs parameterization guide
- **[Circuit Environment](docs/CIRCUIT_ENVIRONMENT.md)** - Waypoint navigation with obstacles
- **[Stairs Height Grid](docs/STAIRS_HEIGHT_GRID_VISUALIZATION.md)** - 5×5 terrain perception mechanics
- **[Stairs Usage](docs/STAIRS_USAGE.md)** - Original stairs environment guide
- **[Walking Humanoid](docs/walking_humanoid.md)** - Destination task walkthrough

### Archive

Historical training reports and bug fixes are available in [docs/archive/](docs/archive/).

---

## 🎓 Training Examples

**Note:** Examples use PowerShell syntax. For Linux/Mac bash, replace `\` with `\` at line ends and activate venv with `source .venv/bin/activate`.

### Walker2d (Baseline)

**PowerShell (Windows):**
```powershell
python scripts/train/train_sb3.py env=walker2d training.total_timesteps=1000000
```

**Bash (Linux/Mac):**
```bash
python scripts/train/train_sb3.py env=walker2d training.total_timesteps=1000000
```

### Humanoid Forward Walking

**PowerShell (Windows):**
```powershell
python scripts/train/train_sb3.py `
  env=humanoid `
  training.total_timesteps=50000000 `
  env.vec_env.n_envs=16
```

**Bash (Linux/Mac):**
```bash
python scripts/30M steps, recommended for learning):**

```powershell
# PowerShell (Windows)
python scripts/train/train_sb3.py `
  env=humanoid_stairs_easy `
  training.total_timesteps=30000000
```

```bash
# Bash (Linux/Mac)
python scripts/train/train_sb3.py \
  env=humanoid_stairs_easy \
  training.total_timesteps=30000000
```

**Evaluation with Custom Environment Parameters:**

```powershell
# PowerShell (Windows) - Note: JSON in single quotes with escaped inner quotes
python scripts/evaluate/evaluate_sb3.py `
  --env_id HumanoidStairsConfigurable-v0 `
  --model_path "outputs/2025-12-06/17-36-50/eval/best_model.zip" `
  --vecnorm_path "outputs/2025-12-06/17-36-50/vecnormalize_final.pkl" `
  --render --deterministic --episodes 5 `
  --env_kwargs '{\"flat_distance_before_stairs\": 2.0, \"num_steps\": 8, \"step_height\": 0.1, \"step_depth\": 0.8, \"end_platform_length\": 5.0, \"stairs_after_top\": false, \"end_with_abyss\": false, \"forward_reward_weight\": 2.0, \"height_reward_weight\": 8.0, \"ctrl_cost_weight\": 0.1, \"contact_cost_weight\": 2e-07, \"healthy_reward\": 15.0, \"step_bonus\": 5.0, \"lateral_penalty_weight\": 0.3, \"check_healthy_z_relative\": true, \"terminate_when_unhealthy\": true, \"healthy_z_range\": [1.0, 2.0]}'
```

```bash
# Bash (Linux/Mac) - JSON in single quotes
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidStairsConfigurable-v0 \
  --model_path "outputs/2025-12-06/17-36-50/eval/best_model.zip" \
  --vecnorm_path "outputs/2025-12-06/17-36-50/vecnormalize_final.pkl" \
  --render --deterministic --episodes 5 \
  --env_kwargs '{"flat_distance_before_stairs": 2.0, "num_steps": 8, "step_height": 0.1, "step_depth": 0.8, "end_platform_length": 5.0, "stairs_after_top": false, "end_with_abyss": false, "forward_reward_weight": 2.0, "height_reward_weight": 8.0, "ctrl_cost_weight": 0.1, "contact_cost_weight": 2e-07, "healthy_reward": 15.0, "step_bonus": 5.0, "lateral_penalty_weight": 0.3, "check_healthy_z_relative": true, "terminate_when_unhealthy": true, "healthy_z_range": [1.0, 2.0]}'ain_sb3.py \
  env=humanoid_stairs_easy \
  training.total_timesteps=5000000 \
  env.vec_env.n_envs=8
```

**Standard Stairs:**
```powershell
python scripts/train/train_sb3.py \
  env=humanoid_stairs \
  Flat Circuit (4 waypoints, 80M steps):**

```powershell
# PowerShell (Windows) - Save output to log
python scripts/train/train_sb3.py `
  env=humanoid_circuit_flat `
  training.total_timesteps=80000000 2>&1 | Tee-Object -FilePath training_circuit.log
```

```bash
# Bash (Linux/Mac) - Save output to log
python scripts/train/train_sb3.py \
  env=humanoid_circuit_flat \
  training.total_timesteps=80000000 2>&1 | tee training_circuit.log
```

**Evaluation with Custom Circuit Configuration:**

```powershell
# PowerShell (Windows)
python scripts/evaluate/evaluate_sb3.py `
  --env_id HumanoidCircuit-v0 `
  --model_path "outputs/2025-12-23/14-37-50/eval/best_model.zip" `
  --vecnorm_path "outputs/2025-12-23/14-37-50/vecnormalize_final.pkl" `
  --render --deterministic --episodes 5 `
  --env_kwargs '{\"waypoints\": [[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]], \"waypoint_reach_threshold\": 1.5, \"stairs\": [], \"terrain_width\": 15.0, \"progress_reward_weight\": 200.0, \"waypoint_bonus\": 150.0, \"circuit_completion_bonus\": 500.0, \"height_reward_weight\": 0.0, \"forward_reward_weight\": 1.0, \"heading_reward_weight\": 2.0, \"balance_reward_weight\": 0.5, \"optimal_speed\": 1.2, \"speed_regulation_weight\": 0.2, \"ctrl_cost_weight\": 0.1, \"contact_cost_weight\": 5e-7, \"healthy_reward\": 5.0, \"terminate_when_unhealthy\": true, \"healthy_z_range\": [0.8, 3.0]}'
```

```bash
# Bash (Linux/Mac)
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidCircuit-v0 \
  --model_path "outputs/2025-12-23/14-37-50/eval/best_model.zip" \
  --vecnorm_path "outputs/2025-12-23/14-37-50/vecnormalize_final.pkl" \
  --render --deterministic --episodes 5 \
  --env_kwargs '{"waypoints": [[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]], "waypoint_reach_threshold": 1.5, "stairs": [], "terrain_width": 15.0, "progress_reward_weight": 200.0, "waypoint_bonus": 150.0, "circuit_completion_bonus": 500.0, "height_reward_weight": 0.0, "forward_reward_weight": 1.0, "heading_reward_weight": 2.0, "balance_reward_weight": 0.5, "optimal_speed": 1.2, "speed_regulation_weight": 0.2, "ctrl_cost_weight": 0.1, "contact_cost_weight": 5e-7, "healthy_reward": 5.0, "terminate_when_unhealthy": true, "healthy_z_range": [0.8, 3.0]}'
```

**Circuit with Stairs (Easy stairs + waypoints):**

```powershell
# PowerShell (Windows)
python scripts/train/train_sb3.py `
  env=humanoid_circuit_easy `
  training.total_timesteps=80000000
```

```bash
# Bash (Linux/Mac)
python scripts/train/train_sb3.py \
  env=humanoid_circuit_easy \
  training.total_timesteps=80000000t (5 waypoints with stairs):**
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

---

## 🧪 Extras: SAC & TorchRL

These alternative approaches are experimental and provided for comparison purposes. The main results use SB3 with PPO.

### SAC Training (Stable-Baselines3)

```bash
# Train Humanoid with SAC
python scripts/train/train_sac.py env=humanoid_sac training.total_timesteps=5000000

# Evaluate SAC model
python scripts/evaluate/evaluate_sac.py \
  --env_id Humanoid-v5 \
  --model_path "outputs_sac/YYYY-MM-DD/HH-MM-SS/checkpoints/model_XXXXXX.zip" \
  --render --deterministic --episodes 5
```

### TorchRL Training

```bash
python scripts/train/train_torchrl.py \
  --env_id Walker2d-v5 \
  --total_frames 1000000 \
  --n_envs 16 \
  --device cpu

# Evaluate TorchRL model
python scripts/evaluate/evaluate_torchrl.py \
  --env_id Walker2d-v5 \
  --model_path "outputs_torchrl/model.pt" \
  --render --episodes 5
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
7. ****Humanoid-v5 (base):** 376 dimensions
   - **Destination:** 378 dims (376 + 2 target vector)
   - **Stairs:** 401 dims (376 + 25 height grid, 5×5, 0.3m spacing)
   - **Circuit:** 406 dims (376 + 25 height grid + 2 waypoint + 1 progress + 2 heading)
   - **Height grid:** 5×5 grid, 0.3m spacing, agent-centered, relative heights
   - **Navigation vectors:** Always relative to agent position (position-invariant learning)
   - See [docs/OBSERVATION_SPACES.md](docs/OBSERVATION_SPACES.md) for complete detailst target
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

### Windows PowerShell Notes

When using `--env_kwargs` with JSON on Windows PowerShell, special escaping is required:

```powershell
# Correct PowerShell syntax for custom environment kwargs
python scripts/evaluate/evaluate_sb3.py --env_id HumanoidCircuit-v0 `
  --model_path "outputs/2025-12-12/09-48-00/eval/best_model.zip" `
  --vecnorm_path "outputs/2025-12-12/09-48-00/vecnormalize_final.pkl" `
  --deterministic --render --episodes 3 `
  --env_kwargs '{\"waypoints\": [[10.0, 0.0], [10.0, 10.0]], \"waypoint_reach_threshold\": 1.0}'
```

**Key PowerShell Tips:**
- Wrap JSON in **single quotes** `'...'`
- Escape inner double quotes with backslashes: `\"`
- Use backticks `` ` `` for line continuation
- Or keep command on one line

---

## 📊 Performance Benchmarks

| Environment | Timesteps | Mean Reward | Observation Dims | Training Time* |
|-------------|-----------|-------------|(376 + 25 grid) | ~5 hours |
| HumanoidStairsHard | 15M | ~7000+ | 401 (376 + 25 grid) | ~15 hours |
| HumanoidCircuitSimple | 20M | ~4000+ | 406 (376 + 25 grid + 5 nav) | ~20 hours |
| HumanoidCircuitComplex | 30M+ | ~3000+ | 406 (376 + 25 grid + 5 nav) | ~30+ hours |

*Approximate on 8-16 parallel environments with CPU

### Observation Space Breakdown

- **Base (376):** Joint positions (22), velocities (23), COM inertia (140), COM velocity (84), actuator forces (17), contact forces (84)
- **Height Grid (+25):** 5×5 terrain height samples, 0.3m spacing, agent-centered
- **Target Vector (+2):** Relative (dx, dy) to destination (Destination env)
- **Navigation (+5):** Waypoint vector (2), progress (1), heading sin/cos (2) (Circuit env) |
| HumanoidStairsHard | 15M | ~7000+ | 401 | ~15 hours |
| HumanoidCircuitSimple | 20M | ~4000+ | 404 (376 + 25 grid + 2 waypoint + 1 progress) | ~20 hours |
| HumanoidCircuitComplex | 30M+ | ~3000+ | 404 | ~30+ hours |

*Approximate on 8-16 parallel environments with CPU

### Key Observations

### Key Observations

- **Height grid** adds 25 observations (5×5 grid, 0.3m spacing, agent-centered)
- **Destination task** uses 2D relative target vector for position-invariant learning
- **Circuit tasks** combine 25-dim height grid + 5-dim navigation (waypoint vector, progress, heading)
- **Navigation components** are always relative to agent (enables generalization)
- **Contact cost** included in Humanoid-v5 and configurable stairs (5e-7 default weight)
- **Resume training** preserves both model weights and VecNormalize statistics

---

## 📊 Best Model Results

The **4 best performing models** for each scenario are documented in [how_to_use.txt](how_to_use.txt), including:
- Exact training commands used
- Evaluation commands with proper `--env_kwargs` configuration
- Model paths and performance metrics

This file serves as a quick reference for reproducing the best results.

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