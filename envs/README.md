# Environments Overview

This repository contains training code for **4 humanoid locomotion environments** with increasing complexity.

---

## 🏃 Built-in Environments (Gymnasium)

These are standard environments from the Gymnasium library used for baseline comparisons.

### 1. **Walker2d-v5**
- **Type:** Built-in (Gymnasium)
- **Difficulty:** ⭐ Easy (2D, introductory)
- **Description:** A simple 2D bipedal walker that learns to move forward on flat ground
- **Observation Dim:** 17
- **Action Dim:** 6
- **Usage:** Introductory task for understanding PPO training
- **Training:** 
  ```bash
  python train.py env=walker2d training.total_timesteps=1000000
  ```

### 2. **Humanoid-v5**
- **Type:** Built-in (Gymnasium)
- **Difficulty:** ⭐⭐ Medium (3D forward locomotion)
- **Description:** Full 3D humanoid learning to walk forward as fast as possible
- **Observation Dim:** 376
- **Action Dim:** 17
- **Goal:** Maximize forward velocity while staying upright
- **Training:**
  ```bash
  python train.py env=humanoid training.total_timesteps=5000000
  ```

---

## 🎯 Custom Environments

These are custom-built environments for specific locomotion challenges.

### 3. **HumanoidStairs-v0** 🪜
- **Type:** Custom
- **Difficulty:** ⭐⭐⭐ Hard (climbing stairs)
- **Description:** Humanoid must learn to climb a staircase
- **Asset:** `envs/assets/humanoid_stairs.xml`
- **Implementation:** `envs/custom/humanoid_stairs.py`
- **Key Features:**
  - 10 stairs, each 0.15m high
  - Stairs span from x=1.5 to x=7.5
  - Requires coordination and balance
  
**Reward Components:**
- ✅ Forward progress (x velocity)
- ✅ Height gained (z position)
- ✅ Survival bonus
- ❌ Control cost penalty

**Observation:** Standard humanoid observations (376-dim)

**Training:**
```bash
python train.py env=humanoid training.total_timesteps=10000000
```

**Evaluation:**
```bash
.\run_2dn.ps1 -EnvId HumanoidStairs-v0 -Best
```

---

### 4. **HumanoidDestination-v0** 🎯
- **Type:** Custom
- **Difficulty:** ⭐⭐⭐ Hard (goal-directed navigation)
- **Description:** Humanoid must walk to a specific target position (x, y)
- **Asset:** `envs/assets/humanoid_destination.xml`
- **Implementation:** `envs/custom/humanoid_destination.py`
- **Key Features:**
  - Flat ground (no obstacles)
  - Default target: (5.0, 0.0)
  - Must learn goal-directed locomotion
  - Configurable target position
  
**Reward Components:**
- ✅ Distance minimization to target
- ✅ Large bonus for reaching target (within 0.5m)
- ✅ Survival bonus
- ❌ Control cost penalty

**Observation:** Standard humanoid observations + relative vector to target (378-dim)

**Training:**
```bash
python train_destination.py training.total_timesteps=2000000
```

**Evaluation:**
```bash
.\run_2dn.ps1 -EnvId HumanoidDestination-v0 -Best
```

---

## 📊 Environment Comparison

| Environment | Type | Obs Dim | Act Dim | Difficulty | Key Challenge |
|-------------|------|---------|---------|------------|---------------|
| Walker2d-v5 | Built-in | 17 | 6 | ⭐ Easy | 2D forward walking |
| Humanoid-v5 | Built-in | 376 | 17 | ⭐⭐ Medium | 3D forward walking |
| HumanoidStairs-v0 | Custom | 376 | 17 | ⭐⭐⭐ Hard | Climbing stairs |
| HumanoidDestination-v0 | Custom | 378 | 17 | ⭐⭐⭐ Hard | Goal navigation |

---

## 🔧 Customizing Environments

### Modify Destination Target

```python
import gymnasium as gym
import envs  # Register custom environments

env = gym.make(
    "HumanoidDestination-v0",
    target_position=(10.0, 5.0),  # Custom target
    reach_reward_weight=20.0,      # Higher reward for reaching
)
```

### Modify Stairs Parameters

Edit `envs/assets/humanoid_stairs.xml` to change:
- Number of stairs
- Step height
- Step width
- Staircase position

---

## 📁 File Structure

```
envs/
├── __init__.py                           # Environment registration
├── README.md                             # This file
├── custom/
│   ├── __init__.py                       # Custom env exports
│   ├── humanoid_stairs.py                # Stairs climbing environment
│   └── humanoid_destination.py           # Goal navigation environment
└── assets/
    ├── humanoid_stairs.xml               # Staircase MuJoCo model
    └── humanoid_destination.xml          # Flat ground MuJoCo model
```

---

## 🚀 Quick Start

1. **Train on stairs:**
   ```bash
   python train.py env=humanoid training.total_timesteps=10000000
   ```

2. **Train destination task:**
   ```bash
   python train_destination.py training.total_timesteps=2000000
   ```

3. **Evaluate trained model:**
   ```bash
   .\run_2dn.ps1 -EnvId HumanoidDestination-v0 -Best
   ```

4. **Compare all environments:**
   ```bash
   python scripts/evaluate/evaluate_all.py
   ```

---

## 📖 References

- **Gymnasium Documentation:** https://gymnasium.farama.org/
- **MuJoCo Physics:** https://mujoco.org/
- **PPO Algorithm:** Proximal Policy Optimization (Schulman et al., 2017)
