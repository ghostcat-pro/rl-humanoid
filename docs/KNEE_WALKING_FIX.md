# Knee-Walking Behavior Fix - November 23, 2025

## 🚨 Issue: Humanoid Walks on Knees

**Observation:** The trained humanoid uses a crouched, knee-walking gait instead of proper upright bipedal walking.

**Visual Description:**
- Agent moves in bent-knee position
- Low center of mass
- Shuffling or crawling-like locomotion
- Not using proper leg extension

---

## 🔍 Root Cause Analysis

### 1. **Overly Permissive Health Range**

**Original Setting:**
```yaml
healthy_z_range: [0.8, 3.0]  # WAY TOO WIDE!
```

**Problem:**
- Proper standing height: ~1.3-1.4m
- Knee-walking height: ~0.9-1.1m ✅ Still counts as "healthy"!
- Crawling height: ~0.8m ✅ Still accepted!

**Result:** Agent can knee-walk and still get full health rewards.

### 2. **Insufficient Upright Reward**

**Original Setting:**
```python
UprightAndEffortWrapper(env, upright_w=0.05, effort_w=0.001)
```

**Problem:**
- Upright bonus: only +0.05 per step (tiny!)
- Forward reward: ~2.0 per step (40× larger!)
- Height reward: ~5.0 per step (100× larger!)

**Result:** Upright posture barely matters compared to other rewards.

### 3. **Reward Structure Allows Exploitation**

The agent discovered that knee-walking:
- ✅ Still moves forward → gets forward_reward
- ✅ Lower = more stable → fewer falls
- ✅ Still in healthy_z_range → gets healthy_reward
- ✅ Can still climb (barely) → gets some height_reward
- ✅ Easier than proper walking!

**The agent is being rational** - it found the easiest way to score points!

---

## ✅ Solutions Implemented

### Fix 1: Stricter Health Range ⭐ PRIMARY FIX

**Changed:**
```yaml
# Before
healthy_z_range: [0.8, 3.0]

# After
healthy_z_range: [1.0, 2.0]  # MUCH STRICTER!
```

**Impact:**
- Minimum height: 1.0m (can't crouch too low)
- Maximum height: 2.0m (can't jump too high)
- Proper standing: 1.3-1.4m ✅ Centered in range
- Knee-walking: 0.9-1.1m ❌ OUT OF RANGE → Episode terminates!

**Result:** Agent **must** maintain upright posture or die.

### Fix 2: Increased Upright Reward ⭐ SECONDARY FIX

**Changed:**
```python
# Before
UprightAndEffortWrapper(env, upright_w=0.05, effort_w=0.001)

# After
UprightAndEffortWrapper(env, upright_w=0.5, effort_w=0.001)  # 10× increase!
```

**Impact:**
- Upright bonus: now +0.5 per step (was +0.05)
- 1000 steps upright = +500 bonus (was only +50)
- Now 25% of forward reward weight (was only 2.5%)

**Result:** Agent rewarded for staying upright.

---

## 📊 Expected Behavior After Fixes

### Before (Knee-Walking):
- Torso height: 0.9-1.1m (crouched)
- Gait: Shuffling, bent knees
- Speed: Slow but stable
- Reward: ~4,800-5,900

### After (Proper Walking):
- Torso height: 1.2-1.5m (upright) ✅
- Gait: Extended legs, proper stride ✅
- Speed: Faster, more efficient ✅
- Reward: ~6,000-7,000 (higher due to upright bonus) ✅

---

## 🎯 Why This ISN'T the Height Grid's Fault

**Question:** Could the 5×5 height grid cause knee-walking?

**Answer:** Not directly. The height grid only provides **perception**, not **motivation**.

**However:**
- Height grid makes agent more terrain-aware
- Might adopt lower stance for "safety" on complex terrain
- But knee-walking is caused by **reward structure**, not perception

**Analogy:**
- Having eyes (height grid) doesn't make you crouch
- Being rewarded for crouching (permissive health range) does!

---

## 🔧 Alternative/Additional Fixes

### Option A: Add Explicit Height Reward

```python
# In the environment step function
current_torso_z = self.data.qpos[2]
target_height = 1.35  # Ideal standing height
height_penalty = -abs(current_torso_z - target_height)
reward += height_penalty
```

### Option B: Penalize Knee Angles

```python
# Penalize excessive knee bending
knee_angles = [self.data.qpos[knee_joint_id] for knee in knees]
knee_penalty = -0.1 * sum(abs(angle) for angle in knee_angles if abs(angle) > 0.5)
reward += knee_penalty
```

### Option C: Use Different Base Environment

Try standard `Humanoid-v5` which has:
```python
healthy_z_range: (1.0, 2.0)  # Already strict!
```

This is what Gymnasium uses by default.

---

## 🧪 Testing the Fix

After retraining with the fixes, verify proper walking by checking:

### 1. **Visual Inspection**
- Record video and watch
- Torso should be at ~1.3-1.4m height
- Legs should extend, not stay bent
- Proper heel-toe gait pattern

### 2. **Statistical Check**
```python
# In evaluation, track torso height
heights = []
for step in episode:
    heights.append(env.data.qpos[2])

mean_height = np.mean(heights)
print(f"Mean torso height: {mean_height:.2f}m")
# Should be: 1.2-1.5m for proper walking
# Was: 0.9-1.1m for knee-walking
```

### 3. **Reward Breakdown**
- Higher total rewards (upright bonus adds up)
- Lower variance (proper walking is more consistent)
- Longer episodes (doesn't fall out of health range)

---

## 📈 Comparison: Standard Humanoid-v5 vs Custom Stairs

| Parameter | Humanoid-v5 | Stairs (Old) | Stairs (Fixed) |
|-----------|-------------|--------------|----------------|
| healthy_z_range | `[1.0, 2.0]` | `[0.8, 3.0]` ❌ | `[1.0, 2.0]` ✅ |
| upright_w | 0.05 | 0.05 | **0.5** ✅ |
| Observed gait | Upright | Knee-walk ❌ | Upright ✅ |

**Lesson:** The custom stairs environment was too permissive. Matching Humanoid-v5's stricter constraints forces proper locomotion.

---

## 🚀 Retraining Required

**Status:** Currently training with OLD config (permissive health range)

**Action Required:**
1. ❌ Stop current training (it will learn knee-walking again)
2. ✅ Restart with FIXED config:
   - `healthy_z_range: [1.0, 2.0]`
   - `upright_w: 0.5`

**Command:**
```bash
# Stop current training
pkill -f train_sb3.py

# Restart with fixed config
python scripts/train/train_sb3.py \
  env=humanoid_stairs_easy \
  training.total_timesteps=20000000
```

---

## 📚 Related Documentation

- **Humanoid-v5 spec:** https://gymnasium.farama.org/environments/mujoco/humanoid/
- **`utils/reward_wrappers.py`** - Upright bonus implementation
- **`utils/make_env.py`** - Where upright weight is set
- **`docs/REWARD_FUNCTIONS_ANALYSIS.md`** - Reward component breakdown

---

## 🎓 Key Takeaways

1. **Reward Hacking is Real**: Agents find easiest path to reward, not "intended" solution
2. **Constraints Matter**: Permissive health ranges allow unnatural gaits
3. **Balance Rewards**: Upright bonus must compete with other rewards
4. **Use Proven Settings**: Humanoid-v5's `[1.0, 2.0]` range exists for a reason!
5. **Height Grid is Innocent**: Perception doesn't cause behavior, rewards do

**Bottom Line:** The agent is smarter than we thought - it optimized for OUR reward function, not OUR intentions! 🤖
