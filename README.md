# RL Humanoid — SB3 + Hydra Skeleton

A minimal, reproducible setup for Gymnasium + Stable-Baselines3 with Hydra configuration.

Train humanoid agents for both **flat ground walking** (Humanoid-v5) and **stair climbing** (HumanoidStairs-v0).

---

## Install
```bash
python -m venv .venv && source .venv/bin/activate # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏃 Training

### Train on Stairs (HumanoidStairs-v0)

Train an agent to climb a 10-step staircase:

```bash
python train.py env=humanoid_stairs
```

**With custom settings:**
```bash
python train.py env=humanoid_stairs \
  training.total_timesteps=10_000_000 \
  env.vec_env.n_envs=16 \
  algo.hyperparams.learning_rate=2.5e-4
```

**Training features:**
- 10-step staircase (each 0.6m deep × 0.15m high)
- Reward shaping for forward progress and height gained
- Automatic timing display (total time, timesteps/sec)
- Checkpoints saved every 250k steps
- TensorBoard logging enabled

---

### Train on Flat Ground (Humanoid-v5)

Train an agent for standard walking:

```bash
python train.py env=humanoid
```

---

### Monitor Training

Watch training progress in real-time with TensorBoard:

```bash
tensorboard --logdir outputs
```

Then open http://localhost:6006 in your browser.

---

## 📊 Evaluation

### Quick Statistics

Get comprehensive performance metrics:

```bash
.venv/bin/python evaluate_stats.py \
  --env_id HumanoidStairs-v0 \
  --model_path outputs/2025-11-05/02-58-25/final_model.zip \
  --vecnorm_path outputs/2025-11-05/02-58-25/vecnormalize_final.pkl \
  --episodes 20 \
  --deterministic
```

**Output includes:**
- Mean reward ± std dev
- Min/Max/Median rewards
- Episode lengths
- Success rate

---

### Live Visualization

Watch your trained agent in action:

```bash
.venv/bin/python evaluate.py \
  --env_id HumanoidStairs-v0 \
  --model_path outputs/2025-11-05/02-58-25/final_model.zip \
  --vecnorm_path outputs/2025-11-05/02-58-25/vecnormalize_final.pkl \
  --episodes 5 \
  --deterministic \
  --render
```

This opens a MuJoCo viewer window showing the humanoid climbing stairs in real-time.

---

### Record Videos

Create MP4 videos of your agent's performance:

```bash
.venv/bin/python evaluate_with_video.py \
  --env_id HumanoidStairs-v0 \
  --model_path outputs/2025-11-05/02-58-25/final_model.zip \
  --vecnorm_path outputs/2025-11-05/02-58-25/vecnormalize_final.pkl \
  --episodes 5 \
  --deterministic \
  --video_dir ./videos/stairs_model
```

Videos are saved to the specified directory as MP4 files.

---

### Evaluate All Models

Automatically evaluate all trained models:

```bash
.venv/bin/python evaluate_all_models.py
```

This will:
- Find all models in the `outputs/` directory
- Auto-detect which environment each was trained on
- Evaluate each for 20 episodes
- Generate a comprehensive `results.txt` report

---

## 📁 Project Structure

```
rl-humanoid/
├── conf/                        # Hydra configuration
│   ├── main.yaml               # Main config
│   ├── env/
│   │   ├── humanoid.yaml       # Flat ground walking
│   │   └── humanoid_stairs.yaml # Stair climbing (NEW)
│   ├── algo/
│   │   └── ppo.yaml            # PPO hyperparameters
│   └── training/
│       ├── default.yaml        # 1M timesteps
│       └── long.yaml           # 5M timesteps
├── envs/                        # Custom environments (NEW)
│   ├── __init__.py             # Environment registration
│   ├── humanoid_stairs.py      # Stairs environment class
│   └── assets/
│       └── humanoid_stairs.xml # MuJoCo physics model
├── utils/                       # Utilities
│   ├── make_env.py             # Environment factory
│   ├── callbacks.py            # Training callbacks
│   └── vecnorm_io.py           # VecNormalize I/O
├── train.py                     # Main training script
├── evaluate.py                  # Basic evaluation
├── evaluate_stats.py            # Statistical evaluation
├── evaluate_with_video.py       # Video recording (NEW)
├── evaluate_all_models.py       # Batch evaluation
└── outputs/                     # Training outputs
    └── YYYY-MM-DD/HH-MM-SS/
        ├── final_model.zip
        ├── vecnormalize_final.pkl
        └── checkpoints/
```

---

## 🎯 Environments

### HumanoidStairs-v0 (Stair Climbing)

**Features:**
- 10-step staircase (each 0.6m deep × 0.15m high)
- Starting platform at x=0
- End platform at x=7.5, height=1.5m
- Observation space: 376 dimensions
- Action space: 17 continuous actions

**Reward Components:**
- Forward progress: 1.25× weight
- Height gained: 2.0× weight
- Staying alive: 5.0 per step
- Control cost: -0.1× penalty

**Training Tips:**
- Requires 5M-10M timesteps for good performance
- Use more parallel environments (16-32) for faster training
- Add entropy bonus for better exploration: `algo.hyperparams.ent_coef=0.005`

---

### Humanoid-v5 (Flat Ground Walking)

**Features:**
- Standard Gymnasium environment
- Flat terrain
- Same observation/action spaces as stairs

**Training:**
- Faster to learn than stairs (1M-2M timesteps)
- Good baseline for comparison

---

## 📚 Documentation

- **[STAIRS_USAGE.md](STAIRS_USAGE.md)** - Detailed stairs environment guide
- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** - Complete evaluation reference
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train on stairs (1M timesteps)
python train.py env=humanoid_stairs

# 3. Monitor training
tensorboard --logdir outputs

# 4. Evaluate your model
.venv/bin/python evaluate_stats.py \
  --env_id HumanoidStairs-v0 \
  --model_path outputs/YYYY-MM-DD/HH-MM-SS/final_model.zip \
  --vecnorm_path outputs/YYYY-MM-DD/HH-MM-SS/vecnormalize_final.pkl \
  --episodes 20 \
  --deterministic

# 5. Watch it live
.venv/bin/python evaluate.py \
  --env_id HumanoidStairs-v0 \
  --model_path outputs/YYYY-MM-DD/HH-MM-SS/final_model.zip \
  --vecnorm_path outputs/YYYY-MM-DD/HH-MM-SS/vecnormalize_final.pkl \
  --episodes 3 \
  --deterministic \
  --render
```

---

## 🔧 Configuration

Training is configured via Hydra. Override any parameter from the command line:

```bash
# Longer training
python train.py env=humanoid_stairs training=long

# More parallel environments
python train.py env=humanoid_stairs env.vec_env.n_envs=16

# Custom learning rate
python train.py env=humanoid_stairs algo.hyperparams.learning_rate=2.5e-4

# Combine multiple overrides
python train.py env=humanoid_stairs \
  training.total_timesteps=10_000_000 \
  env.vec_env.n_envs=16 \
  algo.hyperparams.learning_rate=2.5e-4 \
  algo.hyperparams.ent_coef=0.005
```

---

## 🆚 Environment Comparison

| Feature | Humanoid-v5 | HumanoidStairs-v0 |
|---------|-------------|-------------------|
| Task | Walk on flat ground | Climb stairs |
| Difficulty | Moderate | Hard |
| Training Time | 1M-2M timesteps | 5M-10M timesteps |
| Observation Space | 376 | 376 |
| Action Space | 17 | 17 |
| Terrain | Flat | 10 steps (0.15m high) |
| Reward Function | Standard | Custom (height + forward) |

---

## 💡 Tips

1. **Start with flat ground** - Train on `Humanoid-v5` first to verify your setup
2. **Use checkpoints** - Models are saved every 250k steps for monitoring progress
3. **Compare models** - Use `evaluate_all_models.py` to rank all your experiments
4. **Longer training for stairs** - Stair climbing needs 5M-10M timesteps for good results
5. **Parallel environments** - Increase `n_envs` to speed up training

---

## 🐛 Troubleshooting

**Environment not found error:**
- Make sure scripts import `envs` module (already fixed in evaluate scripts)

**MuJoCo rendering issues:**
```bash
export MUJOCO_GL=glfw
```

**Virtual environment:**
Always activate it before running scripts:
```bash
source .venv/bin/activate
```