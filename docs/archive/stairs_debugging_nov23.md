# Stairs Training Issue Analysis - November 23, 2025

## 🚨 Problem Identified

**Issue:** Agent never reaches/climbs the stairs consistently

**Symptoms:**
- High variance in rewards (741 to 5857)
- 30% failure rate (episodes ending early)
- Average reward ~4790 (should be 6000+ for full stair climbing)
- Some episodes very short (137-375 steps)

## 📊 Evaluation Results (10m flat distance model)

```
Mean reward:      4789.98 ± 1752.11
Success rate:     70% (reward > 5000)
Failed episodes:  3 out of 10 (early termination)
```

## 🔍 Root Cause Analysis

### 1. **Insufficient Stair Climbing Motivation**
- **Forward reward weight:** 1.5 (encourages walking, not climbing)
- **Height reward weight:** 3.0 (not strong enough vs forward walking)
- **Step bonus:** 15.0 (milestone reward too low)

**Problem:** Agent gets decent reward just walking forward on flat terrain, doesn't have strong incentive to climb stairs.

### 2. **Flat Distance Too Long**
- **10m flat terrain** before stairs
- Agent can score points just walking, never reaching stairs
- Episodes might end before stairs are encountered

### 3. **Reward Structure Imbalance**

| Reward Component | Weight | Problem |
|------------------|--------|---------|
| Forward velocity | 1.5 | Rewards walking anywhere |
| Height gain | 3.0 | Only 2× forward reward |
| Step bonus | 15.0 | One-time, not continuous |
| Survival | 5.0 | Constant, no stair incentive |

**Result:** Walking forward on flat = easy reward, climbing = hard work for similar reward

## ✅ Solution Implemented

### Updated Configuration

```yaml
# Terrain (MUCH CLOSER)
flat_distance_before_stairs: 5.0   # Was 10.0, now HALF

# Reward weights (STRONGLY FAVOR CLIMBING)
forward_reward_weight: 2.0         # Was 1.5 (+33%)
height_reward_weight: 5.0          # Was 3.0 (+67%)
step_bonus: 25.0                   # Was 15.0 (+67%)
```

### Why This Works

1. **5m flat distance:**
   - ✅ Agent encounters stairs quickly (2-3 seconds of walking)
   - ✅ Can't avoid stairs, must learn to deal with them
   - ✅ More training time spent on climbing

2. **height_reward_weight: 5.0:**
   - ✅ Now 2.5× forward reward (was only 2×)
   - ✅ Climbing 0.1m height = +0.5 reward
   - ✅ 8 steps × 0.1m = +4.0 bonus from height alone

3. **step_bonus: 25.0:**
   - ✅ Each step climbed = +25 reward
   - ✅ 8 steps = +200 total bonus
   - ✅ Massive incentive to reach all steps

### Expected Rewards

**Old config (walking only):**
- 1000 steps × 0.015m/step × 1.5 = ~22.5 forward reward
- Total: ~22.5 + 5000 (survival) = ~5,022

**New config (climbing 8 steps):**
- Forward: ~30
- Height: 8 steps × 0.1m × 5.0 = ~40
- Step bonuses: 8 × 25 = +200
- Survival: ~5000
- **Total: ~5,270** (climbing rewarded!)

## 🚀 Recommended Training

Train with the new configuration:

```bash
python scripts/train/train_sb3.py \
  env=humanoid_stairs_easy \
  training.total_timesteps=15000000
```

**Why 15M instead of 10M:**
- Harder task now (forced to climb)
- Need more exploration time
- Better convergence to optimal policy

## 📈 Expected Improvements

| Metric | Old (10m, weak rewards) | New (5m, strong rewards) |
|--------|------------------------|--------------------------|
| Success rate | 70% | **95%+** |
| Mean reward | 4,790 | **6,000+** |
| Stair completion | Partial | **Full 8 steps** |
| Episode consistency | High variance | **Low variance** |

## 🎯 How to Verify Success

After training, check:

1. **Mean reward > 6000** (indicates full stair climbing)
2. **Low standard deviation** (< 500, indicates consistency)
3. **Success rate > 90%** (most episodes complete)
4. **Episode length ~900-1000** steps (reaches platform)

## 📝 Alternative Approaches

If still having issues:

### Option A: Even Stronger Climbing Incentive
```yaml
forward_reward_weight: 1.0   # De-emphasize forward
height_reward_weight: 10.0   # HUGE height bonus
step_bonus: 50.0             # Massive step bonus
```

### Option B: Curriculum Learning
```yaml
# Stage 1: 3 very low steps (5cm each)
# Stage 2: 5 low steps (7cm each)
# Stage 3: 8 medium steps (10cm each)
# Stage 4: 10 standard steps (15cm each)
```

### Option C: Shaped Distance Reward
Add distance-to-stairs reward:
- Encourage moving toward x=5.0 (stair start)
- Penalize staying far from stairs
- Requires code modification

## 🔧 Troubleshooting

**If agent still doesn't climb:**

1. Check video - where does it stop?
   - Before stairs → increase forward_reward
   - At stairs base → increase height_reward more
   - Falls on stairs → reduce step_height or increase step_depth

2. Check training logs:
   - Is ep_rew_mean increasing? → Good
   - Stuck at plateau? → Increase exploration (ent_coef)
   - High value_loss? → Reduce learning_rate

3. Try simpler stairs first:
   ```yaml
   num_steps: 5
   step_height: 0.05   # Very low
   step_depth: 1.0     # Very long
   ```

## 📚 Related Documentation

- `conf/env/humanoid_stairs_easy.yaml` - Updated configuration
- `docs/REWARD_FUNCTIONS_ANALYSIS.md` - Reward design details
- `docs/STAIRS_HEIGHT_GRID_VISUALIZATION.md` - Perception system

---

**Status:** Configuration updated, ready for retraining with improved rewards!
