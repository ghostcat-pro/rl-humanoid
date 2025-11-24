# Destination Termination Configuration

## Overview

The `HumanoidDestination-v0` environment now supports configurable termination when the agent reaches its destination. This allows you to control whether episodes end upon success or continue running.

## Parameters

### `terminate_at_destination` (bool)
- **Default:** `True`
- **Description:** Whether to end the episode when the destination is reached
- **Use Cases:**
  - `True`: More efficient training, clearer success signal, realistic behavior
  - `False`: Agent learns to maintain position at destination, longer episodes

### `destination_threshold` (float)
- **Default:** `0.5` meters
- **Description:** Distance threshold to consider destination "reached"
- **Recommendations:**
  - For 10m targets: 0.5m works well (5% tolerance)
  - For 20m+ targets: Consider 1.0m (easier success)
  - For precision tasks: 0.25m (tighter tolerance)

### `target_position` (tuple[float, float])
- **Default:** `(10.0, 0.0)`
- **Description:** Target (x, y) position in world coordinates
- **Examples:**
  - `(5.0, 0.0)`: 5 meters forward (original)
  - `(10.0, 0.0)`: 10 meters forward (current default)
  - `(20.0, 5.0)`: 20m forward, 5m lateral
  - `(15.0, -10.0)`: Diagonal target

## Configuration Examples

### 1. Standard Training (Terminate on Success)

```yaml
# conf/env/humanoid_destination.yaml
make_kwargs:
  target_position: [10.0, 0.0]
  terminate_at_destination: true
  destination_threshold: 0.5
```

**Training:**
```bash
python scripts/train/train_destination.py
```

**Result:** Episodes end when agent reaches within 0.5m of (10, 0), enabling efficient training.

---

### 2. Stabilization Training (Continue After Success)

```yaml
make_kwargs:
  target_position: [10.0, 0.0]
  terminate_at_destination: false
  destination_threshold: 0.5
```

**Training:**
```bash
python scripts/train/train_destination.py \
  env.make_kwargs.terminate_at_destination=false
```

**Result:** Episodes run for full 1000 steps, agent learns to stay at destination.

---

### 3. Long Distance Challenge

```yaml
make_kwargs:
  target_position: [25.0, 0.0]
  terminate_at_destination: true
  destination_threshold: 1.0  # Larger threshold for far targets
```

**Training:**
```bash
python scripts/train/train_destination.py \
  env.make_kwargs.target_position=[25.0,0.0] \
  env.make_kwargs.destination_threshold=1.0 \
  training.total_timesteps=30000000  # More training needed
```

**Result:** 25m target with 1m tolerance, requires longer training.

---

### 4. Diagonal Navigation

```yaml
make_kwargs:
  target_position: [15.0, 10.0]
  terminate_at_destination: true
  destination_threshold: 0.75
```

**Training:**
```bash
python scripts/train/train_destination.py \
  env.make_kwargs.target_position=[15.0,10.0] \
  env.make_kwargs.destination_threshold=0.75
```

**Result:** Agent must navigate diagonally (18m total distance).

---

### 5. Precision Task

```yaml
make_kwargs:
  target_position: [5.0, 0.0]
  terminate_at_destination: true
  destination_threshold: 0.25  # Tight tolerance
```

**Training:**
```bash
python scripts/train/train_destination.py \
  env.make_kwargs.target_position=[5.0,0.0] \
  env.make_kwargs.destination_threshold=0.25
```

**Result:** Agent must reach very close to target (precision navigation).

---

## Training Recommendations

### Distance vs. Training Duration

| Target Distance | Recommended Steps | Threshold | Notes |
|-----------------|------------------|-----------|-------|
| 5m (close) | 10-15M | 0.5m | Quick learning, good for testing |
| 10m (default) | 20M | 0.5m | Balanced challenge |
| 15m (far) | 25M | 0.75m | Requires good locomotion |
| 20m+ (very far) | 30M+ | 1.0m | Expert level, needs stable gait |

### Termination Strategy

**When to use `terminate_at_destination=true`:**
✅ Efficient training (fewer wasted steps)  
✅ Clear success metrics (episode length)  
✅ Curriculum learning (gradually increase distance)  
✅ Real-world deployment (task ends when complete)  

**When to use `terminate_at_destination=false`:**
✅ Learning to stabilize at goal  
✅ Multi-objective tasks (reach + maintain)  
✅ Debugging reward shaping  
✅ Collecting longer trajectories  

---

## Code Implementation

### Python API

```python
import gymnasium as gym

# Terminate on success (efficient)
env = gym.make(
    "HumanoidDestination-v0",
    target_position=(15.0, 5.0),
    terminate_at_destination=True,
    destination_threshold=0.5,
)

# Continue after success (stabilization)
env = gym.make(
    "HumanoidDestination-v0",
    target_position=(10.0, 0.0),
    terminate_at_destination=False,
    destination_threshold=0.5,
)
```

### Checking Success in Info Dict

```python
obs, info = env.reset()

for step in range(1000):
    action = model.predict(obs)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check if destination was reached
    if info['reached_destination']:
        print(f"Success! Reached destination at step {step}")
        print(f"Distance: {info['distance_to_target']:.2f}m")
    
    if terminated:
        if info.get('reached_destination', False):
            print("Episode ended: SUCCESS")
        else:
            print("Episode ended: Health failure")
        break
```

---

## Evaluation Examples

### Evaluate with Termination

```bash
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidDestination-v0 \
  --model_path outputs/.../final_model.zip \
  --vecnorm_path outputs/.../vecnormalize_final.pkl \
  --episodes 10 \
  --deterministic
```

### Evaluate WITHOUT Termination (see full behavior)

```bash
# Create temporary config
python scripts/evaluate/evaluate_sb3.py \
  --env_id HumanoidDestination-v0 \
  --model_path outputs/.../final_model.zip \
  --vecnorm_path outputs/.../vecnormalize_final.pkl \
  --episodes 5 \
  --render \
  --env_kwargs '{"terminate_at_destination": false}'
```

---

## Testing

Run the test script to verify termination behavior:

```bash
python test_destination_termination.py
```

Expected output:
- Test 1: Episode ends when destination reached (~100-300 steps)
- Test 2: Episode runs full 1000 steps even if destination reached
- Test 3: Shows default config values

---

## Troubleshooting

**Issue:** Episodes end too quickly
- **Solution:** Increase `destination_threshold` or set `terminate_at_destination=false`

**Issue:** Agent learns to "touch and go" (briefly reach then leave)
- **Solution:** This is expected with termination ON. If you want stabilization, use `terminate_at_destination=false`

**Issue:** Training is inefficient (episodes run too long)
- **Solution:** Enable `terminate_at_destination=true` to end episodes on success

**Issue:** Success rate is 0% with far targets
- **Solution:** Increase `destination_threshold` or train longer. For 20m+ targets, use threshold ≥1.0m

---

## Migration from Previous Version

**Old behavior:** Episodes always ran for 1000 steps regardless of success

**New default:** Episodes end when destination reached (more efficient)

**To restore old behavior:**
```yaml
make_kwargs:
  terminate_at_destination: false
```

Or via command line:
```bash
python scripts/train/train_destination.py \
  env.make_kwargs.terminate_at_destination=false
```

---

## Summary

The configurable termination feature provides flexibility for different training objectives:

- **Default (terminate ON):** Best for most use cases, efficient training
- **Terminate OFF:** Useful for stabilization and long trajectory collection
- **Threshold:** Adjust based on target distance and precision requirements

Use Hydra overrides to experiment with different configurations without modifying files! 🎯
