# Circuit Navigation Environment

## Overview

`HumanoidCircuit-v0` is a complex navigation task where the humanoid must sequentially visit a series of waypoints while navigating configurable obstacles (stairs). This environment combines goal-directed navigation with terrain traversal.

## Key Features

- **Sequential Waypoint Navigation**: Must visit waypoints in order
- **Configurable Obstacles**: Stairs can be placed between waypoints
- **Terrain Perception**: 5x5 height grid around the agent
- **Relative Coordinates**: Position-agnostic observations
- **Progress Tracking**: Completion percentage and waypoint count
- **Visual Markers**: Waypoints visible as colored boxes in simulation

## Observation Space

**404 dimensions total:**
- **376**: Base Humanoid-v5 observations (joint positions, velocities, etc.)
- **25**: Height grid (5x5 terrain perception around agent)
- **2**: Relative vector to current waypoint (x, y)
- **1**: Progress (waypoints_reached / total_waypoints)

## Action Space

**17 continuous actions**: Same as Humanoid-v5 (joint torques)

## Reward Function

```python
total_reward = (
    progress_reward +           # Progress toward current waypoint
    waypoint_reward +          # Bonus for reaching waypoint
    height_reward +            # Bonus for maintaining height
    forward_reward +           # Velocity in forward direction
    - control_cost            # Penalty for large actions
    - contact_cost            # Penalty for harsh contacts
)
```

### Reward Components

| Component | Formula | Default Weight | Purpose |
|-----------|---------|----------------|---------|
| **Progress** | `(prev_dist - curr_dist) × weight` | 100 | Reward getting closer to waypoint |
| **Waypoint Bonus** | `weight` (when reached) | 50 | Large bonus for completing waypoint |
| **Height** | `height × weight` | 1.0 | Encourage upright posture |
| **Forward** | `1.25 × (dx/dt)` | 1.0 | Reward forward velocity |
| **Control Cost** | `0.1 × ‖action‖²` | 0.1 | Penalize large actions |
| **Contact Cost** | `5e-7 × ‖cfrc_ext‖²` | 5e-7 | Penalize harsh contacts |

## Configuration Parameters

### Waypoints Configuration

```yaml
waypoints:
  - [10.0, 0.0]   # First waypoint at (x=10, y=0)
  - [20.0, 0.0]   # Second waypoint
  - [30.0, 5.0]   # Third waypoint (with turn)
  - [40.0, 5.0]   # Final waypoint
```

- **Format**: List of [x, y] coordinates
- **Order**: Agent must visit in sequence
- **Threshold**: Configurable reach distance (default 1.0m)

### Stairs Configuration

```yaml
stairs:
  - [x_start, num_steps, step_height, step_depth]
  # Example:
  - [8.0, 5, 0.15, 0.6]   # Stairs from x=8 to x=11
  - [28.0, 7, 0.12, 0.8]  # Stairs from x=28 to x=33.6
```

**Stair Parameters:**
- `x_start`: Start position (m)
- `num_steps`: Number of steps
- `step_height`: Height per step (m, typical: 0.10-0.20)
- `step_depth`: Depth per step (m, typical: 0.6-1.0)

### Reward Weights

```yaml
# Tune these for different behaviors
progress_reward_weight: 100      # Higher = more focus on navigation
waypoint_bonus: 50              # Higher = stronger waypoint completion incentive
height_reward_weight: 1.0       # Height maintenance importance
forward_reward_weight: 1.0      # Forward velocity importance
ctrl_cost_weight: 0.1          # Action smoothness penalty
contact_cost_weight: 5e-7      # Contact smoothness penalty
```

## Pre-configured Scenarios

### 1. Simple Circuit (`humanoid_circuit_simple.yaml`)

**Purpose**: Basic sequential navigation with moderate obstacles

```yaml
waypoints:
  - [10.0, 0.0]
  - [20.0, 0.0]
  - [30.0, 0.0]
  - [40.0, 0.0]

stairs:
  - [8.0, 5, 0.15, 0.6]    # Before waypoint 1
  - [28.0, 5, 0.15, 0.6]   # Before waypoint 3

rewards:
  progress: 100
  waypoint_bonus: 50
```

**Difficulty**: Medium  
**Training**: Good starting point for learning combined task

---

### 2. Complex Circuit (`humanoid_circuit_complex.yaml`)

**Purpose**: Advanced navigation with turns and varied obstacles

```yaml
waypoints:
  - [10.0, 0.0]
  - [20.0, 0.0]
  - [30.0, 5.0]    # Turn required
  - [40.0, 5.0]
  - [40.0, -5.0]   # Another turn
  - [50.0, -5.0]

stairs:
  - [8.0, 5, 0.15, 0.6]     # Standard stairs
  - [18.0, 7, 0.12, 0.8]    # Longer, gentler stairs
  - [38.0, 4, 0.20, 0.5]    # Fewer, taller steps

rewards:
  progress: 120            # Higher for more waypoints
  waypoint_bonus: 100      # Stronger completion incentive
```

**Difficulty**: Hard  
**Training**: Requires good navigation + climbing skills

---

### 3. Easy Circuit (`humanoid_circuit_easy.yaml`)

**Purpose**: Curriculum learning starting point

```yaml
waypoints:
  - [10.0, 0.0]
  - [20.0, 0.0]
  - [30.0, 0.0]

stairs:
  - [8.0, 4, 0.10, 0.8]    # Easy stairs (gentle slope)

rewards:
  progress: 150            # Very high navigation reward
  waypoint_bonus: 30       # Lower bonus (focus on navigation)
  ctrl_cost: 0.05          # Lower penalty for exploration
  contact_cost: 2e-7       # Lower penalty for exploration
```

**Difficulty**: Easy  
**Training**: First stage of curriculum

---

### 4. Flat Circuit (`humanoid_circuit_flat.yaml`)

**Purpose**: Pure navigation without obstacles

```yaml
waypoints:
  - [10.0, 0.0]
  - [10.0, 10.0]   # Square pattern
  - [0.0, 10.0]
  - [0.0, 0.0]     # Return to start

stairs: []         # No obstacles

rewards:
  progress: 200              # Very high navigation focus
  waypoint_bonus: 50
  height_reward_weight: 0.0  # No height reward needed
```

**Difficulty**: Easy  
**Training**: Pre-training for navigation skills only

---

## Episode Termination

**Episode ends when:**
- Agent falls (z < 1.0)
- Unhealthy state (flipping, bad orientation)
- Max steps reached (2000 timesteps)
- **Success**: All waypoints visited

## Training Strategy

### Curriculum Learning Approach

```bash
# Stage 1: Learn navigation (no obstacles)
python scripts/train/train_sb3.py env=humanoid_circuit_flat training.total_timesteps=5000000

# Stage 2: Add simple obstacle
python scripts/train/train_sb3.py env=humanoid_circuit_easy training.total_timesteps=10000000

# Stage 3: Standard circuit
python scripts/train/train_sb3.py env=humanoid_circuit_simple training.total_timesteps=15000000

# Stage 4: Complex multi-obstacle
python scripts/train/train_sb3.py env=humanoid_circuit_complex training.total_timesteps=20000000
```

### Transfer Learning Option

```python
# Pre-train navigation skills
train(env="HumanoidDestination-v0", timesteps=10M)

# Pre-train climbing skills  
train(env="HumanoidStairsConfigurable-v0", timesteps=10M)

# Fine-tune on combined task
train(env="HumanoidCircuit-v0", timesteps=5M, 
      load_policy="destination+stairs_merged")
```

## Usage Examples

### Basic Usage

```python
import gymnasium as gym
import envs

# Use pre-configured scenario
env = gym.make("HumanoidCircuit-v0", config_name="simple")
obs, info = env.reset()

# Training loop
for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if info['waypoints_reached'] == len(env._waypoints):
        print("All waypoints reached!")
        break
```

### Custom Configuration

```python
# Define custom circuit
env = gym.make(
    "HumanoidCircuit-v0",
    waypoints=[(10, 0), (20, 5), (30, 0)],
    stairs=[
        (8.0, 5, 0.15, 0.6),   # Before waypoint 1
        (18.0, 6, 0.12, 0.7)   # Before waypoint 2
    ],
    waypoint_reach_threshold=1.5,
    progress_reward_weight=120,
    waypoint_bonus=80,
)
```

### Monitoring Progress

```python
obs, info = env.reset()

for step in range(max_steps):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Track progress
    print(f"Waypoint {info['current_waypoint_index']}/{len(env._waypoints)}")
    print(f"Distance: {info['distance_to_waypoint']:.2f}m")
    print(f"Progress: {info['waypoints_reached']/len(env._waypoints)*100:.1f}%")
    
    # Check waypoint completion
    if 'waypoint_reached' in info and info['waypoint_reached']:
        print(f"Reached waypoint {info['current_waypoint_index']-1}!")
```

## Performance Metrics

### Success Criteria

- **Navigation**: Reach all waypoints in sequence
- **Efficiency**: Minimize time/steps to completion
- **Quality**: Maintain upright posture, smooth movements
- **Robustness**: Handle varied terrain and waypoint patterns

### Evaluation Metrics

```python
# After episode
total_waypoints = len(env._waypoints)
waypoints_reached = info['waypoints_reached']
completion_rate = waypoints_reached / total_waypoints

# Success if all waypoints reached
success = completion_rate == 1.0

# Performance metrics
episode_length = step_count
avg_reward_per_step = total_reward / episode_length
time_to_complete = episode_length if success else float('inf')
```

## Comparison with Other Environments

| Environment | Task | Obstacles | Waypoints | Observation Dims |
|-------------|------|-----------|-----------|------------------|
| Humanoid-v5 | Forward locomotion | None | None | 376 |
| HumanoidDestination-v0 | Single goal | None | 1 | 378 |
| HumanoidStairs-v0 | Climb stairs | Fixed stairs | None | 401 |
| **HumanoidCircuit-v0** | **Sequential goals** | **Configurable** | **Multiple** | **404** |

## Debugging Tips

### Waypoint Not Advancing

- Check `waypoint_reach_threshold` (may be too small)
- Verify waypoint positions are reachable
- Monitor `distance_to_waypoint` in info dict

### Agent Ignoring Waypoints

- Increase `progress_reward_weight`
- Increase `waypoint_bonus`
- Reduce `forward_reward_weight` (may be going straight)

### Struggling with Stairs

- Use curriculum: start with flat → easy → standard
- Pre-train on HumanoidStairs-v0
- Adjust stairs difficulty (height, depth, num_steps)

### Observation Issues

```python
# Check observation composition
obs_base = obs[:376]           # Base Humanoid
obs_height = obs[376:401]      # Height grid (5x5)
obs_waypoint = obs[401:403]    # Vector to waypoint
obs_progress = obs[403]        # Completion percentage

print(f"Vector to waypoint: ({obs_waypoint[0]:.2f}, {obs_waypoint[1]:.2f})")
print(f"Progress: {obs_progress*100:.1f}%")
```

## Related Documentation

- [Configurable Stairs Environment](CONFIGURABLE_STAIRS.md)
- [Reward Function Analysis](REWARD_FUNCTIONS_ANALYSIS.md)
- [Main README](../README.md)

## Citation

If you use this environment, please cite:

```bibtex
@misc{humanoid_circuit,
  title={HumanoidCircuit: Sequential Waypoint Navigation with Configurable Obstacles},
  author={Your Name},
  year={2025},
  note={Custom Gymnasium environment for humanoid locomotion research}
}
```
