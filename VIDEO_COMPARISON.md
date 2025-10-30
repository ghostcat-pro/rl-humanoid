# Humanoid-v5 Model Comparison

## Video Recordings Created

Videos have been successfully generated for both models using xvfb (virtual framebuffer) rendering.

### 100M Iterations Model (Latest - 21-33-51)
Location: `./videos/`
- `eval-Humanoid-v5-step-0-to-step-1000.mp4` (1.1M) - Episode 1
- `eval-Humanoid-v5-step-1370-to-step-2370.mp4` (304K) - Episodes 2-3

**Performance (3 episodes):**
- Mean reward: 4603.77 ± 3743.43
- Episode 1: 9750.68 (excellent - full 1000 steps)
- Episode 2: 3103.59 (fell after 370 steps)
- Episode 3: 957.06 (fell after 142 steps)

### 50M Iterations Model (17-47-25)
Location: `./videos/50M/`
- `eval-Humanoid-v5-step-0-to-step-1000.mp4` (1.1M) - Episode 1
- `eval-Humanoid-v5-step-1000-to-step-2000.mp4` (1.1M) - Episode 2
- `eval-Humanoid-v5-step-2000-to-step-3000.mp4` (1.1M) - Episode 3

**Performance (3 episodes):**
- Mean reward: 9776.07 ± 91.91
- Episode 1: 9866.46 (perfect)
- Episode 2: 9811.76 (perfect)
- Episode 3: 9649.99 (perfect)

## Analysis

### 50M Model (Better Performance)
- ✅ **Very consistent** - all 3 episodes achieved near-perfect scores
- ✅ **Stable locomotion** - completed all 1000 steps in each episode
- ✅ **Low variance** (±91.91) - highly reliable
- Mean reward: **9776.07**

### 100M Model (More Variable)
- ⚠️ **Inconsistent** - 1 perfect episode, 2 failures
- ⚠️ **Less stable** - tendency to fall in some episodes
- ⚠️ **High variance** (±3743.43) - unreliable performance
- Mean reward: **4603.77**

## Conclusion

The 50M iteration model appears to be **significantly better** than the 100M model. This suggests:

1. **Overfitting** - The 100M model may have overfit to the training distribution
2. **Overtraining** - Training past 50M iterations degraded performance
3. **Policy degradation** - Extended training didn't improve the policy

## Recommendation

**Use the 50M model** (`outputs/2025-10-28/17-47-25/final_model.zip`) for deployment, as it shows:
- More consistent behavior
- Higher success rate
- Better generalization

## How to View Videos

You can now watch the videos to see the humanoid in action:

```bash
# Open with default video player
xdg-open videos/eval-Humanoid-v5-step-0-to-step-1000.mp4
xdg-open videos/50M/eval-Humanoid-v5-step-0-to-step-1000.mp4

# Or use VLC
vlc videos/eval-Humanoid-v5-step-0-to-step-1000.mp4
```

## Generate More Videos

To create more evaluation videos:

```bash
# For 100M model
./run_2d_display.sh --use-final-model --env-id Humanoid-v5 --episodes 10

# For 50M model
./run_2d_display.sh --use-final-model --env-id Humanoid-v5 --run outputs/2025-10-28/17-47-25 --video-dir ./videos/50M --episodes 10

# For any specific run
./run_2d_display.sh --run outputs/2025-10-28/[timestamp] --video-dir ./videos/custom
```
