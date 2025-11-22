# Configurable Stairs Environment

## Overview

`HumanoidStairsConfigurable-v0` is a highly flexible stairs climbing environment that allows extensive customization of terrain parameters. Unlike the original `HumanoidStairs-v0` (which has fixed stairs), this environment generates terrain dynamically based on configuration parameters.

## Features

### Terrain Configuration Parameters

1. **`flat_distance_before_stairs`** (float, default: 20.0)
   - Distance in meters of flat terrain before the first step
   - Allows the agent to build walking momentum before climbing
   - Range: 0.0+ (but at least 5.0 recommended)

2. **`num_steps`** (int, default: 10)
   - Number of stairs going upward
   - Each step is a discrete height increase
   - Range: 1-50 (practical limit)

3. **`step_height`** (float, default: 0.15)
   - Height of each step in meters
   - Standard stairs: 0.15m (15cm)
   - Easy stairs: 0.10m (10cm)
   - Hard stairs: 0.20m (20cm)
   - Range: 0.05 - 0.30 (practical limits)

4. **`step_depth`** (float, default: 0.6)
   - Depth/length of each step in meters
   - Standard stairs: 0.6m
   - Longer steps (0.8m) are easier
   - Shorter steps (0.4m) are harder
   - Range: 0.2 - 1.0

5. **`end_platform_length`** (float, default: 5.0)
   - Length of flat platform at the top of stairs
   - Provides a safe landing zone
   - Set to 0.0 if using `end_with_abyss`
   - Range: 0.0 - 10.0

6. **`stairs_after_top`** (bool, default: False)
   - If True, stairs continue downward after the top platform
   - Creates an up-then-down scenario
   - Requires `num_steps_down` parameter

7. **`num_steps_down`** (int, default: 5)
   - Number of steps going downward (only if `stairs_after_top=True`)
   - Range: 1-50

8. **`end_with_abyss`** (bool, default: False)
   - If True, no end platform (agent must stop at top step)
   - Challenging: agent must learn to stop climbing
   - Overrides `end_platform_length`

### Reward Configuration Parameters

- **`forward_reward_weight`** (float, default: 1.25): Weight for forward velocity
- **`height_reward_weight`** (float, default: 2.0): Weight for gaining height
- **`ctrl_cost_weight`** (float, default: 0.1): Penalty for control effort
- **`healthy_reward`** (float, default: 5.0): Reward for staying upright
- **`step_bonus`** (float, default: 10.0): Bonus for reaching new steps

### Health Configuration

- **`terminate_when_unhealthy`** (bool, default: True): End episode on fall
- **`healthy_z_range`** (tuple, default: (0.8, 3.0)): Valid height range

## Observation Space

**Dimension: 401** (376 base humanoid + 25 height grid)

The observation includes:
- Standard humanoid state (376 dims): positions, velocities, forces, etc.
- **5×5 height grid** (25 dims): Relative terrain heights around the agent
  - Grid spacing: 0.3m
  - Covers 1.2m × 1.2m area
  - Heights relative to agent's current z-position
  - Provides terrain awareness for stepping

## Pre-configured Scenarios

### 1. Standard Stairs (`humanoid_stairs_configurable.yaml`)
- 20m flat → 10 steps (15cm × 60cm) → 5m platform
- Balanced difficulty for general training

### 2. Easy Stairs (`humanoid_stairs_easy.yaml`)
- 20m flat → 8 steps (10cm × 80cm) → 5m platform
- Shallower, longer steps
- Higher step bonus (15.0)
- Good for initial learning

### 3. Hard Stairs (`humanoid_stairs_hard.yaml`)
- 15m flat → 15 steps (20cm × 40cm) → 3m platform
- Steeper, shorter steps
- More challenging for advanced agents

### 4. Short Approach (`humanoid_stairs_short.yaml`)
- **5m** flat → 10 steps (15cm × 60cm) → 5m platform
- Quick iterations for debugging/testing

### 5. Abyss Ending (`humanoid_stairs_abyss.yaml`)
- 20m flat → 10 steps → **NO PLATFORM**
- Agent must learn to stop at the top
- Lower forward reward (1.0) to discourage rushing

### 6. Up and Down (`humanoid_stairs_updown.yaml`)
- 20m flat → 10 steps up → 3m platform → 8 steps down
- Tests both climbing and descending
- Lower height reward (1.5) since height is temporary

### 7. Tiny Steps (`humanoid_stairs_tiny.yaml`)
- 10m flat → 20 steps (7.5cm × 30cm)
- Many small steps for fine motor control
- Lower step bonus (5.0) due to more steps

## Usage

### Via Hydra Configuration

```bash
# Standard configuration
python scripts/train/train_sb3.py env=humanoid_stairs_configurable

# Easy stairs
python scripts/train/train_sb3.py env=humanoid_stairs_easy

# Hard stairs
python scripts/train/train_sb3.py env=humanoid_stairs_hard

# Abyss ending
python scripts/train/train_sb3.py env=humanoid_stairs_abyss

# Up and down
python scripts/train/train_sb3.py env=humanoid_stairs_updown

# Tiny steps
python scripts/train/train_sb3.py env=humanoid_stairs_tiny
```

### Programmatically

```python
import gymnasium as gym
import envs  # Register custom environments

# Create with custom parameters
env = gym.make(
    "HumanoidStairsConfigurable-v0",
    flat_distance_before_stairs=15.0,
    num_steps=12,
    step_height=0.18,
    step_depth=0.5,
    end_platform_length=4.0,
    stairs_after_top=False,
    end_with_abyss=False,
)

# Use environment
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Creating Custom Configurations

Create a new YAML file in `conf/env/`:

```yaml
# @package _global_

name: HumanoidStairsConfigurable-v0

make_kwargs:
  render_mode: null
  
  # Your custom terrain parameters
  flat_distance_before_stairs: 25.0
  num_steps: 12
  step_height: 0.12
  step_depth: 0.7
  end_platform_length: 6.0
  stairs_after_top: false
  end_with_abyss: false
  
  # Your custom reward weights
  forward_reward_weight: 1.5
  height_reward_weight: 2.5
  ctrl_cost_weight: 0.08
  healthy_reward: 5.0
  step_bonus: 12.0

vec_env:
  n_envs: 8
  start_method: spawn
  monitor: true
```

Then use it:
```bash
python scripts/train/train_sb3.py env=your_custom_config
```

## Testing

Run the test script to verify all configurations work:

```bash
python test_configurable_stairs.py
```

This will test all 6 pre-configured scenarios and verify:
- Environment creation
- Observation space dimensions
- Reward calculation
- Step mechanics

## Implementation Details

### Dynamic XML Generation

The environment generates MuJoCo XML files **dynamically** based on parameters:
- No need for separate XML files for each configuration
- Platform and stair geometries calculated programmatically
- Temporary XML file created on initialization and cleaned up on deletion

### Terrain Height Calculation

The `_get_terrain_height_at(x, y)` method provides analytical terrain heights:
- Starting platform: height = 0
- Stairs going up: height = step_index × step_height
- End platform: height = num_steps × step_height
- Stairs going down: height decreases by step_height per step

### Height Grid Perception

5×5 grid sampled around agent:
- Center point: agent's current position
- Grid spacing: 0.3m
- Returns **relative heights** (terrain_height - agent_z)
- Provides lookahead for stepping decisions

## Curriculum Learning Suggestions

1. **Progressive Difficulty**:
   - Start: `humanoid_stairs_easy`
   - Middle: `humanoid_stairs_configurable`
   - Advanced: `humanoid_stairs_hard`

2. **Transfer Learning**:
   - Pre-train on flat terrain (Humanoid-v5)
   - Fine-tune on `humanoid_stairs_short` (quick approach)
   - Transfer to `humanoid_stairs_configurable`

3. **Exploration**:
   - Train on random configurations by varying parameters
   - Use domain randomization for robustness

## Comparison with Original Stairs Environment

| Feature | HumanoidStairs-v0 | HumanoidStairsConfigurable-v0 |
|---------|-------------------|-------------------------------|
| Terrain | Fixed (XML file) | Dynamic (parameterized) |
| Flat distance | 20m (fixed) | Configurable |
| Number of steps | 10 (fixed) | Configurable |
| Step dimensions | Fixed | Configurable |
| Abyss ending | No | Optional |
| Stairs down | No | Optional |
| Configuration files | 1 | 7 pre-made + unlimited custom |
| Use case | Standard benchmark | Experimentation & curriculum learning |

## Tips for Best Results

1. **Start with easy configurations**: Use `humanoid_stairs_easy` for initial training
2. **Tune rewards carefully**: Height reward vs forward reward balance is critical
3. **Monitor step progress**: Use TensorBoard to track `max_step_reached` metric
4. **Adjust step bonus**: Higher bonus encourages climbing, lower encourages smooth motion
5. **Use abyss ending carefully**: Very challenging - agent must learn precise stopping
6. **Experiment with flat distance**: More distance = better walking, less stair practice

## Future Extensions

Potential enhancements:
- Uneven step heights (randomized)
- Variable step widths
- Gaps between steps
- Moving stairs
- Angled/curved stairs
- Multi-stage stairs with landings
