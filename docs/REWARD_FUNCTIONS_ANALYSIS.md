# Reward Functions - Detailed Analysis

## Overview

Each environment uses different reward functions optimized for their specific tasks. This document provides detailed analysis and comparison with the **standard Gymnasium Humanoid-v5** environment.

---

## 0. Humanoid-v5 (Standard Gymnasium Environment)

**Source:** Gymnasium MuJoCo environments (official baseline)

### Reward Components

```python
reward = healthy_reward + forward_reward - ctrl_cost - contact_cost
```

### Breakdown

#### 1. **Healthy Reward** (survival)
```python
healthy_reward = 5.0 if is_healthy else 0.0
```
- **Weight:** 5.0
- **Type:** Binary reward
- **Condition:** `1.0 < z_position < 2.0` (stricter than custom envs)
- **Purpose:** Stay upright and alive
- **Note:** In v5, only given when healthy (bug fixed from v4)

#### 2. **Forward Reward** (displacement-based)
```python
forward_reward = 1.25 * (dx / dt)
```
- **Weight:** 1.25 (default `forward_reward_weight`)
- **Type:** **Displacement per timestep** (not velocity!)
- **Calculation:**
  - `dx = x_after - x_before` (displacement)
  - `dt = frame_skip × frametime = 5 × 0.003 = 0.015`
  - So: `forward_reward = 1.25 × (displacement / 0.015)`
- **Example:** Moving 0.1m forward → `1.25 × (0.1/0.015) = 8.33`
- **Purpose:** Encourages fast forward movement
- **Key Difference:** Custom stairs use **velocity** (vx), not displacement/dt

#### 3. **Control Cost** (efficiency)
```python
ctrl_cost = 0.1 * np.sum(np.square(action))
```
- **Weight:** 0.1 (default `ctrl_cost_weight`)
- **Same across all environments**

#### 4. **Contact Cost** (smoothness) ⭐
```python
contact_cost = 5e-7 * np.clip(np.sum(np.square(cfrc_ext)), -inf, 10.0)
```
- **Weight:** 5e-7 (default `contact_cost_weight`)
- **Type:** L2 penalty on external contact forces
- **Source:** `self.data.cfrc_ext` (78 elements, 13 bodies × 6 force components)
- **Clipping:** Maximum value clamped to 10.0
- **Purpose:** 
  - Penalizes hard impacts/collisions
  - Encourages smooth, controlled movement
  - Reduces joint stress
  - Important for sim-to-real transfer
- **Note:** This was 0 in v4 due to a bug, restored in v5

### Typical Reward Magnitudes

| Component | Typical Range | Notes |
|-----------|--------------|-------|
| Healthy | 5.0 | Constant when alive |
| Forward | 0 to +15 | ~8.33 per 0.1m displacement |
| Ctrl Cost | -0.1 to -2 | Varies with action magnitude |
| Contact Cost | -0.001 to -0.01 | Very small, penalizes impacts |
| **Total** | **~3 to ~18** | Per timestep when walking |

### Key Characteristics

✅ **Displacement-based forward reward** (encourages speed)  
✅ **Contact cost included** (encourages smooth movement)  
✅ **Stricter health range** (1.0-2.0 vs 0.8-3.0)  
❌ **No height reward** (flat terrain only)  
❌ **No milestone rewards** (no stairs/targets)  

---

## 1. HumanoidStairs-v0 (Original Stairs)

**Location:** `envs/custom/humanoid_stairs.py`

### Reward Components

```python
reward = forward_reward + height_reward + healthy_reward + step_reward + stuck_penalty - ctrl_cost
```

**Note:** No contact cost (unlike standard Humanoid-v5)

### Breakdown

#### 1. **Forward Reward** (velocity-based)
```python
forward_reward = 1.25 * xy_velocity[0]
```
- **Weight:** 1.25 (hardcoded)
- **Type:** Velocity reward
- **Range:** Can be negative if moving backward
- **Purpose:** Encourages forward movement

#### 2. **Height Reward** (gain-based)
```python
height_gained = z_position - prev_z_position
height_reward = 2.0 * max(0, height_gained)
```
- **Weight:** 2.0 (hardcoded)
- **Type:** Delta reward (height gain per timestep)
- **Range:** [0, ∞) - only positive gains rewarded
- **Purpose:** Encourages climbing

#### 3. **Healthy Reward** (survival)
```python
healthy_reward = 5.0 if is_healthy else 0.0
```
- **Weight:** 5.0 (hardcoded)
- **Type:** Binary reward
- **Range:** {0.0, 5.0}
- **Condition:** `0.8 < z_position < 3.0`
- **Purpose:** Stay upright and alive

#### 4. **Step Reward** (milestone-based)
```python
if current_step > max_step_reached:
    step_reward = 10.0 * (current_step - max_step_reached)
else:
    step_reward = 0.0
```
- **Weight:** 10.0 per new step (hardcoded)
- **Type:** Sparse milestone reward
- **Range:** {0, 10, 20, 30, ...}
- **Purpose:** Reward reaching new stairs

#### 5. **Stuck Penalty** (anti-stagnation)
```python
if not terminated and np.linalg.norm(xy_velocity) < 0.1:
    stuck_penalty = -0.1
else:
    stuck_penalty = 0.0
```
- **Weight:** -0.1 (hardcoded)
- **Type:** Penalty for low velocity
- **Threshold:** velocity < 0.1 m/s
- **Purpose:** Discourage standing still

#### 6. **Control Cost** (efficiency)
```python
ctrl_cost = 0.1 * np.sum(np.square(action))
```
- **Weight:** 0.1 (hardcoded)
- **Type:** L2 penalty on actions
- **Range:** [0, ∞)
- **Purpose:** Encourage smooth, efficient movements

### Typical Reward Magnitudes

| Component | Typical Range | Notes |
|-----------|--------------|-------|
| Forward | -2 to +5 | ~1.25 per m/s forward |
| Height | 0 to +1 | ~2.0 per 0.5m climbed per step |
| Healthy | 5.0 | Constant when alive |
| Step | 0 to 100+ | 10 per new step reached |
| Stuck | -0.1 or 0 | Small penalty |
| Ctrl Cost | -0.1 to -2 | Varies with action magnitude |
| **Total** | **~3 to ~15** | Per timestep when climbing |

---

## 2. HumanoidDestination-v0

**Location:** `envs/custom/humanoid_destination.py`

### Reward Components

```python
reward = progress_reward + reach_reward + healthy_reward - ctrl_cost
```

### Breakdown

#### 1. **Progress Reward** (distance improvement)
```python
prev_dist = np.linalg.norm(target_position - prev_xy_position)
dist = np.linalg.norm(target_position - xy_position)
progress_reward = (prev_dist - dist) * 1.0 * 100  # distance_reward_weight=1.0
```
- **Weight:** 100.0 (1.0 × 100 scale factor)
- **Type:** Delta reward (distance improvement)
- **Range:** Can be negative if moving away from target
- **Purpose:** Reward getting closer to destination

#### 2. **Reach Reward** (goal bonus)
```python
if dist < 0.5:  # Within 0.5 meters
    reach_reward = 10.0
else:
    reach_reward = 0.0
```
- **Weight:** 10.0
- **Type:** Sparse bonus
- **Threshold:** 0.5m from target
- **Purpose:** Large reward for reaching destination

#### 3. **Healthy Reward** (survival)
```python
healthy_reward = 5.0 if is_healthy else 0.0
```
- **Weight:** 5.0
- **Same as stairs environment**

#### 4. **Control Cost** (efficiency)
```python
ctrl_cost = 0.1 * np.sum(np.square(action))
```
- **Weight:** 0.1
- **Same as stairs environment**

### Typical Reward Magnitudes

| Component | Typical Range | Notes |
|-----------|--------------|-------|
| Progress | -10 to +10 | Depends on movement toward/away from target |
| Reach | 0 or 10 | Bonus when within 0.5m |
| Healthy | 5.0 | Constant when alive |
| Ctrl Cost | -0.1 to -2 | Same as stairs |
| **Total** | **~3 to ~25** | Higher when near target |

---

## 3. HumanoidStairsConfigurable-v0

**Location:** `envs/custom/humanoid_stairs_configurable.py`

### Reward Components

```python
reward = forward_reward + height_reward + healthy_reward + step_reward - ctrl_cost - contact_cost
```

**Key Feature:** All reward weights are **configurable parameters**!

### Breakdown

#### 1. **Forward Reward** (velocity-based)
```python
forward_reward = forward_reward_weight * xy_velocity[0]
```
- **Default Weight:** 1.25
- **Configurable:** YES ✓
- **Same as original stairs**

#### 2. **Height Reward** (gain-based)
```python
height_gained = z_position - prev_z_position
height_reward = height_reward_weight * max(0, height_gained)
```
- **Default Weight:** 2.0
- **Configurable:** YES ✓
- **Same as original stairs**

#### 3. **Healthy Reward** (survival)
```python
healthy_reward = healthy_reward if is_healthy else 0.0
```
- **Default Weight:** 5.0
- **Configurable:** YES ✓

#### 4. **Step Reward** (milestone-based)
```python
if current_step > max_step_reached:
    step_reward = step_bonus * (current_step - max_step_reached)
else:
    step_reward = 0.0
```
- **Default Weight:** 10.0
- **Configurable:** YES ✓
- **Note:** NO stuck penalty (removed for flexibility)

#### 5. **Control Cost** (efficiency)
```python
ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))
```
- **Default Weight:** 0.1
- **Configurable:** YES ✓

#### 6. **Contact Cost** (smoothness) ⭐ **NEW!**
```python
contact_cost = contact_cost_weight * np.sum(np.square(self.data.cfrc_ext))
```
- **Default Weight:** 5e-7 (same as Humanoid-v5)
- **Configurable:** YES ✓
- **Type:** L2 penalty on external contact forces
- **Purpose:** 
  - Penalizes hard landings on stairs
  - Encourages smooth, controlled stepping
  - Reduces violent impacts
  - Critical for stairs climbing quality
- **Benefits:**
  - More natural gaits
  - Better sim-to-real transfer
  - Prevents reward hacking through aggressive movements
  - Smoother foot-stair contact patterns

### Configuration Variants

| Config | Forward | Height | Healthy | Step Bonus | Ctrl Cost | Contact Cost | Notes |
|--------|---------|--------|---------|------------|-----------|--------------|-------|
| **standard** | 1.25 | 2.0 | 5.0 | 10.0 | 0.1 | **5e-7** | Balanced, same as Humanoid-v5 |
| **easy** | 1.5 | 3.0 | 5.0 | 15.0 | 0.05 | **2e-7** | Higher rewards, lower costs for exploration |
| **hard** | 1.25 | 2.0 | 5.0 | 10.0 | 0.1 | **1e-6** | 2x contact cost for smoother movement |
| **abyss** | 1.0 | 2.0 | 5.0 | 10.0 | 0.1 | **5e-7** | Lower forward (don't rush off edge!) |
| **updown** | 1.25 | 1.5 | 5.0 | 10.0 | 0.1 | **5e-7** | Lower height (goes back down) |
| **tiny** | 1.5 | 2.0 | 5.0 | 5.0 | 0.1 | **5e-7** | Lower step bonus (more frequent steps) |

---

## Key Differences Summary

### 1. Reward Philosophy

| Environment | Primary Goal | Reward Type | Forward Type | Contact Cost |
|-------------|--------------|-------------|--------------|-------------|
| **Humanoid-v5** | Walk forward fast | Dense velocity + survival | Displacement/dt | ✅ Yes (5e-7) |
| **Stairs (original)** | Climb stairs | Mixed (dense velocity + sparse milestones) | Velocity (vx) | ❌ No |
| **Destination** | Reach target | Dense progress + sparse goal | Distance delta | ❌ No |
| **Stairs (configurable)** | Climb stairs (flexible) | Same as original but tunable | Velocity (vx) | ✅ Yes (5e-7, configurable) |

### 2. Forward Reward Type Comparison

**Humanoid-v5 (Standard):**
- Rewards **displacement per timestep**: `1.25 × (dx / 0.015)`
- Higher magnitude (dx/dt can be large)
- Encourages speed and efficiency
- Example: 0.1m displacement → reward ≈ 8.33

**Stairs environments:**
- Reward **velocity directly**: `1.25 × vx`
- Lower magnitude (velocity typically 0-2 m/s)
- Smoother reward signal
- Example: 1.0 m/s velocity → reward = 1.25

**Destination environment:**
- Rewards **distance improvement**: `100 × (prev_dist - curr_dist)`
- Can be negative if moving away
- Directly tied to goal progress

### 3. Configurability

| Environment | Configurable Rewards? | Contact Cost |
|-------------|----------------------|-------------|
| Humanoid-v5 | ✅ Yes (via make kwargs) | ✅ Yes (5e-7 default) |
| HumanoidStairs-v0 | ❌ No (hardcoded) | ❌ No |
| HumanoidDestination-v0 | ❌ No (hardcoded) | ❌ No |
| HumanoidStairsConfigurable-v0 | ✅ Yes (all weights tunable) | ✅ Yes (5e-7 default, fully tunable) |

### 4. Unique Features

| Environment | Unique Reward Components |
|-------------|-------------------------|
| **Humanoid-v5 (Standard)** | • Contact cost (5e-7)<br>• Displacement-based forward reward<br>• Stricter health range (1.0-2.0) |
| **Stairs (original)** | • Stuck penalty (-0.1)<br>• Step milestone reward<br>• Height gain reward<br>• No contact cost |
| **Destination** | • Progress reward (distance delta)<br>• Reach bonus (goal proximity)<br>• No contact cost |
| **Stairs (configurable)** | • All stairs features<br>• **Contact cost (configurable)**<br>• No stuck penalty<br>• Configurable everything |

**Contact Cost Impact on Stairs:**
- Penalizes hard foot-stair impacts
- Encourages smooth stepping transitions
- Reduces jerky/violent movements
- Important for learning natural gaits
- Default 5e-7 (same as Humanoid-v5), but tunable per scenario

---

## Reward Engineering Considerations

### When to Use Each Environment

1. **HumanoidStairs-v0** (Original)
   - Fixed benchmark for comparing algorithms
   - Standard stair climbing task
   - Has stuck penalty to prevent standing still

2. **HumanoidDestination-v0**
   - Goal-directed navigation
   - When you need the agent to reach a specific point
   - Dense reward signal (good for early learning)

3. **HumanoidStairsConfigurable-v0**
   - Experimentation with reward weights
   - Curriculum learning (easy → hard)
   - Testing different stair configurations
   - No stuck penalty (more exploration freedom)

### Reward Tuning Tips

#### For Faster Learning:
- ↑ Increase `step_bonus` (more motivation to climb)
- ↑ Increase `height_reward_weight` (value climbing more)
- ↓ Decrease `ctrl_cost_weight` (allow more exploration)
- ↓ Decrease `contact_cost_weight` (less penalty for rough movements)

#### For Better Quality Movement:
- ↓ Decrease `forward_reward_weight` (don't rush)
- ↑ Increase `ctrl_cost_weight` (smoother actions)
- ↑ Increase `contact_cost_weight` (penalize hard impacts more)
- ↓ Decrease `step_bonus` (less jerky milestone-seeking)

#### For Sim-to-Real Transfer:
- ↑ Increase `contact_cost_weight` to 1e-6 or higher (enforce smooth contact)
- Balance with `ctrl_cost_weight` for energy efficiency
- Use contact cost to prevent policies that work in sim but damage real hardware

#### Contact Cost Configuration by Scenario:
- **Early exploration:** `2e-7` (low penalty, allow learning)
- **Standard training:** `5e-7` (same as Humanoid-v5)
- **Quality refinement:** `1e-6` (2x penalty, enforce smoothness)
- **Sim-to-real prep:** `2e-6` or higher (very smooth movements)

#### For Specific Scenarios:
- **Abyss ending:** Lower `forward_reward_weight` to 1.0 (prevent overshooting)
- **Up and down:** Lower `height_reward_weight` to 1.5 (height is temporary)
- **Tiny steps:** Lower `step_bonus` to 5.0 (more frequent rewards)

---

## Mathematical Comparison

### Expected Reward per Episode

Assumptions: 1000 timesteps, successful climb to top (10 steps)

#### Humanoid-v5 (Standard)
```
Healthy:   5.0 * 1000 = 5000
Forward:   1.25 * (total_displacement / 0.015) ≈ 1.25 * (6m / 0.015) = 500
Ctrl:      -0.1 * ~0.5 * 1000 = -50
Contact:   -5e-7 * ~1000 * 1000 = -0.5
────────────────────────────────────
Total:     ~5450
```

#### Stairs (Original) - No Contact Cost
```
Forward:   1.25 * avg_velocity * 1000 ≈ 1.25 * 0.3 * 1000 = 375
Height:    2.0 * 1.5 total_height = 3.0
Healthy:   5.0 * 1000 = 5000
Steps:     10.0 * 10 steps = 100
Stuck:     -0.1 * ~100 times = -10
Ctrl:      -0.1 * ~0.5 * 1000 = -50
────────────────────────────────────
Total:     ~5418
```

#### Stairs (Configurable) - With Contact Cost
```
Forward:   1.25 * avg_velocity * 1000 ≈ 1.25 * 0.3 * 1000 = 375
Height:    2.0 * 1.5 total_height = 3.0
Healthy:   5.0 * 1000 = 5000
Steps:     10.0 * 10 steps = 100
Ctrl:      -0.1 * ~0.5 * 1000 = -50
Contact:   -5e-7 * ~1000 * 1000 = -0.5
────────────────────────────────────
Total:     ~5427.5
```

#### Destination (5m target)
```
Progress:  100 * 5m improvement = 500
Reach:     10.0 (once at target) = 10
Healthy:   5.0 * 1000 = 5000
Ctrl:      -0.1 * ~0.5 * 1000 = -50
────────────────────────────────────
Total:     ~5460
```

**All environments have similar total rewards! Contact cost is small but important for movement quality.**

---

## Recommendations

### For Your Current Work:

1. **Keep Stairs-v0** for baseline comparison (no contact cost)
2. **Use Configurable** for experimentation:
   - Start with `humanoid_stairs_easy` (lower contact cost 2e-7)
   - Progress to `humanoid_stairs_configurable` (standard 5e-7)
   - Challenge with `humanoid_stairs_hard` (higher contact cost 1e-6)
   - Challenge with `humanoid_stairs_abyss`

3. **Contact Cost Strategy:**
   ```yaml
   # Early training (allow exploration)
   contact_cost_weight: 2e-7  # Low penalty
   
   # Standard training (balanced)
   contact_cost_weight: 5e-7  # Same as Humanoid-v5
   
   # Fine-tuning (smooth movement)
   contact_cost_weight: 1e-6  # 2x penalty for quality
   ```

4. **Reward Tuning Strategy:**
   ```yaml
   # Early training (encourage exploration)
   forward_reward_weight: 1.5
   height_reward_weight: 3.0
   step_bonus: 15.0
   ctrl_cost_weight: 0.05
   contact_cost_weight: 2e-7  # Low
   
   # Fine-tuning (improve quality)
   forward_reward_weight: 1.0
   height_reward_weight: 2.0
   step_bonus: 10.0
   ctrl_cost_weight: 0.15
   contact_cost_weight: 1e-6  # Higher for smoothness
   ```

5. **Comparison with Standard Humanoid-v5:**
   - Monitor `info["cost_contact"]` in TensorBoard
   - Compare contact patterns between flat walking and stairs
   - Stairs should have similar or slightly lower contact costs (controlled stepping)
   - If contact cost is very high, agent is landing too hard

### Potential Issues to Watch:

1. **Reward Hacking:**
   - Agent might learn to "bounce" to gain height reward repeatedly
   - Solution: Cap height reward or use cumulative height only

2. **Local Minima:**
   - Agent might get stuck on early steps if step_bonus too high
   - Solution: Balance forward_reward with step_bonus

3. **Rushing:**
   - High forward_reward might cause agent to run and fall
   - Solution: Increase ctrl_cost or decrease forward_reward

4. **Stuck Behavior:**
   - Without stuck penalty (configurable env), might stand still
   - Solution: Monitor and add back if needed, or rely on forward_reward

5. **Hard Landings (Contact Cost Related):**
   - Agent might learn to jump/stomp on stairs if contact_cost_weight too low
   - **Symptoms:** Very high `info["cost_contact"]` values (>0.01)
   - **Solution:** Increase `contact_cost_weight` to 1e-6 or higher
   - **Monitor:** Track contact cost in TensorBoard alongside rewards

6. **Over-Cautious Movement:**
   - Contact cost too high might make agent move too slowly/carefully
   - **Symptoms:** Very low velocity, agent barely moving
   - **Solution:** Decrease `contact_cost_weight` or increase `forward_reward_weight`
   - **Balance:** Contact cost should be ~0.1-1% of total reward magnitude
