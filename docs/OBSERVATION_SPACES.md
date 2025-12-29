# Observation Spaces - Complete Documentation

## Overview

This document provides detailed information about the observation space dimensions and components for all custom environments, including how they differ from the standard Gymnasium Humanoid-v5 environment.

---

## Base Humanoid Observation (376 dimensions)

All custom environments inherit from the standard Humanoid-v5 observation structure:

### Standard Components (376 total)

```python
observation = [
    position[2:],            # 22 dims: Joint positions (qpos), excluding global x,y
    velocity,                # 23 dims: Joint velocities (qvel)
    com_inertia,            # 140 dims: Center of mass inertia (14 bodies × 10 components)
    com_velocity,           # 84 dims: Center of mass velocity (14 bodies × 6 components)
    actuator_forces,        # 17 dims: Actuator forces (qfrc_actuator)
    external_contact_forces # 84 dims: External contact forces (14 bodies × 6 components)
]
```

**Breakdown:**
- **Position (22):** `qpos[2:]` - excludes global (x, y) position
  - Root z-position (height)
  - Root orientation (quaternion: w, x, y, z)
  - Joint positions for all body parts
  
- **Velocity (23):** `qvel` - all velocity components
  - Root linear velocity (3D)
  - Root angular velocity (3D)
  - Joint velocities

- **COM Inertia (140):** `cinert` - 14 bodies × 10 components each
  - Mass and rotational inertia information

- **COM Velocity (84):** `cvel` - 14 bodies × 6 components each
  - Linear and angular velocities of each body part

- **Actuator Forces (17):** `qfrc_actuator`
  - Forces applied by motors/actuators

- **External Contact Forces (84):** `cfrc_ext` - 14 bodies × 6 components each
  - Forces from external contacts (ground, stairs, etc.)

**Note:** Global x,y position is excluded to encourage learning position-invariant policies.

---

## 1. HumanoidDestination-v0

**Total Dimensions:** 378

### Structure

```python
observation = [
    # Base humanoid observation
    position[2:],            # 22 dims
    velocity,                # 23 dims
    com_inertia,            # 140 dims
    com_velocity,           # 84 dims
    actuator_forces,        # 17 dims
    external_contact_forces, # 84 dims
    
    # Navigation-specific additions
    vector_to_target        # 2 dims: (dx, dy) relative to target
]
```

### Additional Components

| Component | Dimensions | Description | Range |
|-----------|------------|-------------|-------|
| **vector_to_target** | 2 | Relative vector from agent to target position (10m, 0m) | (-inf, +inf) |

**Purpose:** The relative vector allows the agent to navigate to the target regardless of its current position. This enables position-invariant learning and generalization.

**Implementation:**
```python
current_xy = position[0:2]  # Before excluding global position
vector_to_target = target_position - current_xy  # [10.0, 0.0] - [x, y]
```

---

## 2. HumanoidStairs-v0 (Original)

**Total Dimensions:** 401

### Structure

```python
observation = [
    # Base humanoid observation
    position[2:],            # 22 dims
    velocity,                # 23 dims
    com_inertia,            # 140 dims
    com_velocity,           # 84 dims
    actuator_forces,        # 17 dims
    external_contact_forces, # 84 dims
    
    # Terrain perception addition
    height_grid             # 25 dims: 5×5 grid of terrain heights
]
```

### Additional Components

| Component | Dimensions | Description | Range |
|-----------|------------|-------------|-------|
| **height_grid** | 25 | 5×5 grid of terrain heights sampled around agent | (-inf, +inf) |

### Height Grid Details

**Configuration:**
- **Grid Size:** 5×5 (25 sample points)
- **Sample Spacing:** 0.3m between points
- **Coverage Area:** 1.2m × 1.2m (agent at center)
- **Height Encoding:** Relative to agent's current z-position

**Grid Layout (Top View):**
```
    • • • • •     
    • • • • •     
    • • A • •     ← Agent (A) at center
    • • • • •     
    • • • • •     

Coverage: 1.2m × 1.2m
Spacing: 0.3m
Total points: 25
```

**Sampling Pattern:**
```python
for i in range(5):  # x-direction
    for j in range(5):  # y-direction
        dx = -0.6 + i * 0.3  # Offset from agent
        dy = -0.6 + j * 0.3
        sample_x = current_x + dx
        sample_y = current_y + dy
        terrain_height = get_terrain_height_at(sample_x, sample_y)
        relative_height = terrain_height - current_z
        height_grid.append(relative_height)
```

**Purpose:** 
- Local terrain awareness
- Perceive upcoming stairs
- Plan foot placement
- Symmetric perception (agent centered)

---

## 3. HumanoidStairsConfigurable-v0

**Total Dimensions:** 401

### Structure

Same as HumanoidStairs-v0:

```python
observation = [
    # Base humanoid observation
    position[2:],            # 22 dims
    velocity,                # 23 dims
    com_inertia,            # 140 dims
    com_velocity,           # 84 dims
    actuator_forces,        # 17 dims
    external_contact_forces, # 84 dims
    
    # Terrain perception addition
    height_grid             # 25 dims: 5×5 grid of terrain heights
]
```

**Identical to HumanoidStairs-v0** - The difference is in configurability of terrain and rewards, not observation space.

### Height Grid Configuration

Same 5×5 grid structure, but terrain heights vary based on configuration:

| Configuration | Step Height | Step Depth | Num Steps | Grid Samples |
|---------------|-------------|------------|-----------|--------------|
| **stairs_easy** | 0.10m | 0.6m | 8 | 25 (5×5) |
| **stairs** (standard) | 0.15m | 0.6m | 10 | 25 (5×5) |
| **stairs_hard** | 0.18m | 0.6m | 12 | 25 (5×5) |
| **stairs_tiny** | 0.075m | 0.6m | 20 | 25 (5×5) |
| **stairs_abyss** | 0.15m | 0.6m | 10 | 25 (5×5) |
| **stairs_updown** | 0.15m | 0.6m | 10 up + down | 25 (5×5) |

**Note:** Grid structure remains constant; only the terrain geometry changes.

---

## 4. HumanoidCircuit-v0

**Total Dimensions:** 406

### Structure

```python
observation = [
    # Base humanoid observation
    position[2:],            # 22 dims
    velocity,                # 23 dims
    com_inertia,            # 140 dims
    com_velocity,           # 84 dims
    actuator_forces,        # 17 dims
    external_contact_forces, # 84 dims
    
    # Terrain perception
    height_grid,            # 25 dims: 5×5 grid of terrain heights
    
    # Navigation components
    vector_to_waypoint,     # 2 dims: (dx, dy) to current waypoint
    waypoint_progress,      # 1 dim: fraction of waypoints reached
    heading_error_obs       # 2 dims: (sin(error), cos(error))
]
```

### Additional Components

| Component | Dimensions | Description | Range |
|-----------|------------|-------------|-------|
| **height_grid** | 25 | 5×5 terrain height grid (same as stairs) | (-inf, +inf) |
| **vector_to_waypoint** | 2 | Relative vector to current active waypoint | (-inf, +inf) |
| **waypoint_progress** | 1 | Normalized progress: waypoints_reached / total_waypoints | [0.0, 1.0] |
| **heading_error_obs** | 2 | Heading error encoded as (sin, cos) to avoid discontinuity | [-1.0, +1.0] |

### Component Details

#### Height Grid
- Same 5×5 structure as stairs environments
- Samples terrain including multiple stair sections
- Agent-centered for symmetric perception

#### Vector to Waypoint
```python
current_xy = position[0:2]
vector_to_waypoint = current_waypoint - current_xy  # Dynamic, updates when waypoint reached
```
- **Dynamic:** Updates to next waypoint when current is reached
- **Sequential:** Must visit waypoints in order
- **Purpose:** Direct navigation signal

#### Waypoint Progress
```python
waypoint_progress = waypoints_reached / len(waypoints)  # e.g., 2/6 = 0.333
```
- **Normalized:** Always in [0, 1] range
- **Purpose:** Provides context of overall circuit completion
- **Example:** For 6-waypoint circuit:
  - Start: 0.0
  - After 1st waypoint: 0.167
  - After 3rd waypoint: 0.5
  - Completed: 1.0

#### Heading Error Observation
```python
# Compute heading error
yaw = arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))  # From quaternion
target_dir = arctan2(waypoint_dy, waypoint_dx)
heading_error = target_dir - yaw  # Wrapped to [-π, π]

# Encode as sin/cos to avoid discontinuity
heading_error_obs = [sin(heading_error), cos(heading_error)]
```
- **Purpose:** Agent knows if facing toward waypoint
- **Encoding:** Sin/cos avoids discontinuity at ±π
- **Interpretation:**
  - [0, 1]: Facing directly at waypoint
  - [1, 0]: Facing 90° right of waypoint
  - [0, -1]: Facing away from waypoint
  - [-1, 0]: Facing 90° left of waypoint

### Circuit Configuration Variants

| Config | Waypoints | Stairs | Height Grid | Heading | Progress |
|--------|-----------|--------|-------------|---------|----------|
| **flat** | 3 | 0 | All zero | ✓ | ✓ |
| **simple** | 4 | 2 sections | Variable | ✓ | ✓ |
| **easy** | 3 | 1 gentle | Variable | ✓ | ✓ |
| **complex** | 6 | 3 varied | Variable | ✓ | ✓ |
| **custom** | 5 | Configurable | Variable | ✓ | ✓ |

**Note:** Observation space size is constant (406) across all circuit variants.

---

## Comparison Table

| Environment | Total Dims | Base | Height Grid | Target/Waypoint | Progress | Heading | Notes |
|-------------|-----------|------|-------------|-----------------|----------|---------|-------|
| **Humanoid-v5** | 376 | 376 | - | - | - | - | Standard baseline |
| **HumanoidDestination-v0** | 378 | 376 | - | 2 (target) | - | - | Navigation only |
| **HumanoidStairs-v0** | 401 | 376 | 25 | - | - | - | Terrain perception |
| **HumanoidStairsConfigurable-v0** | 401 | 376 | 25 | - | - | - | Same as stairs |
| **HumanoidCircuit-v0** | 406 | 376 | 25 | 2 (waypoint) | 1 | 2 | Multi-objective |

---

## Design Principles

### 1. Position Invariance
All environments exclude global (x, y) position from observations:
- **Benefits:** 
  - Policies generalize across starting positions
  - Reduces observation space dimensionality
  - Focuses learning on relative movements

- **Implementation:**
  ```python
  if exclude_current_positions_from_observation:
      position = position[2:]  # Remove qpos[0:2]
  ```

### 2. Relative Encoding
Navigation information is always relative to agent:
- **Target vectors:** Relative distance, not absolute coordinates
- **Height grid:** Heights relative to agent's current z-position
- **Waypoint vectors:** Relative to agent's current position

**Benefits:**
- Translation invariance
- Easier learning (local frame of reference)
- Generalizes to different starting positions

### 3. Continuous Representations
Avoid discontinuities in observations:
- **Heading error:** Encoded as sin/cos, not raw angle
  - Raw angle: Discontinuity at ±π (-3.14 → +3.14)
  - Sin/cos: Smooth continuous representation
  
- **Height grid:** Relative heights smoothly update as agent moves

### 4. Symmetric Perception
Height grid is always centered on agent:
- **5×5 grid:** Agent at exact center (2, 2) in grid coordinates
- **Coverage:** Equal perception in all directions (1.2m radius)
- **Purpose:** Symmetric perception aids learning balanced gaits

---

## Implementation Details

### Height Grid Sampling

**Terrain Height Function:**
```python
def _get_terrain_height_at(self, x, y):
    """
    Get terrain height at a specific (x, y) position.
    Returns height based on stair geometry.
    """
    if x < stairs_start_x:
        return 0.0  # Flat platform
    elif x >= stairs_end_x:
        return max_height  # Top platform
    else:
        # On stairs - calculate step
        step_index = int((x - stairs_start_x) / step_depth)
        return step_index * step_height
```

**Grid Assembly:**
```python
def _get_height_grid(self):
    heights = []
    grid_half_size = (5 - 1) * 0.3 / 2  # 0.6m
    
    for i in range(5):
        for j in range(5):
            # Calculate sample position
            dx = -grid_half_size + i * 0.3
            dy = -grid_half_size + j * 0.3
            sample_x = current_x + dx
            sample_y = current_y + dy
            
            # Get terrain height
            terrain_h = self._get_terrain_height_at(sample_x, sample_y)
            
            # Store relative height
            relative_h = terrain_h - current_z
            heights.append(relative_h)
    
    return np.array(heights, dtype=np.float64)
```

### Waypoint Vector (Circuit)

```python
# Get current position (before excluding from obs)
current_xy = position[0:2]

# Current active waypoint
current_waypoint = waypoints[current_waypoint_index]

# Compute relative vector
vector_to_waypoint = current_waypoint - current_xy

# Example:
# current_xy = [5.0, 0.2]
# current_waypoint = [10.0, 0.0]
# vector_to_waypoint = [5.0, -0.2]
```

---

## Usage Considerations

### 1. Observation Scaling
- Some components have very different magnitudes
- Consider normalization via `VecNormalize` wrapper
- Monitor observation statistics during training

### 2. Height Grid Interpretation
- Negative values: Terrain below agent (agent on raised surface)
- Positive values: Terrain above agent (upcoming stair/obstacle)
- Zero values: Terrain at same height as agent

**Example - Approaching Stairs:**
```
Grid at x=19.5 (before stairs at x=20.0):
Front of grid shows positive heights (stairs ahead)
Back of grid shows zeros or negatives (flat ground behind)

[-0.1, -0.1, -0.1, -0.1, -0.1]  ← Behind agent (flat)
[-0.1, -0.1, -0.1, -0.1, -0.1]
[ 0.0,  0.0,  0.0,  0.0,  0.0]  ← At agent level
[ 0.1,  0.1,  0.1,  0.1,  0.1]  ← Ahead (stairs starting)
[ 0.2,  0.2,  0.2,  0.2,  0.2]  ← Further ahead (higher stairs)
```

### 3. Navigation Vector Magnitude
- **Destination:** Starts at ~10m (distance to [10, 0])
- **Circuit waypoints:** Varies based on configuration
  - Typically 5-15m between waypoints
  - Gets smaller as agent approaches

### 4. Heading Error Range
- Sin/cos encoding: Each component in [-1, 1]
- Perfect alignment: [0, 1] (sin=0, cos=1)
- Perpendicular: [±1, 0]
- Opposite: [0, -1]

---

## Training Implications

### 1. Network Architecture
Observation sizes suggest network capacity:
- **378 dims (Destination):** Relatively simple, standard networks work
- **401 dims (Stairs):** Need sufficient capacity for height grid processing
- **406 dims (Circuit):** Most complex, may benefit from larger networks

### 2. Feature Importance
Different components have different importance:
- **Base humanoid (376):** Essential for all tasks
- **Height grid (25):** Critical for stairs, less for flat terrain
- **Navigation vectors (2-3):** High importance for goal-directed tasks

### 3. Curriculum Learning
Observation complexity suggests curriculum:
1. Start with Destination (378 dims, simpler task)
2. Progress to Stairs (401 dims, terrain perception)
3. Advance to Circuit (406 dims, multi-objective)

### 4. VecNormalize Importance
- **Recommendation:** Always use `VecNormalize` wrapper
- **Reason:** Different observation components have vastly different scales
  - Position: ~0-2 meters
  - Velocity: ~0-5 m/s
  - Forces: ~0-1000+ N
  - Height grid: ~-3 to +3 meters
  - Navigation vectors: ~0-15 meters

---

## Debugging Observations

### Common Issues

**1. Observation Contains NaN/Inf:**
```python
# Check in info dict
assert not np.any(np.isnan(obs))
assert not np.any(np.isinf(obs))
```

**2. Height Grid All Zeros:**
- Check if agent is outside terrain bounds
- Verify `_get_terrain_height_at()` implementation

**3. Navigation Vector Not Updating:**
- Verify waypoint reached logic
- Check threshold for waypoint completion

**4. Heading Error Jumps:**
- Should be smooth with sin/cos encoding
- If jumping, check quaternion extraction

### Visualization
```python
# Print observation breakdown
obs_parts = {
    "position": obs[0:22],
    "velocity": obs[22:45],
    "com_inertia": obs[45:185],
    "com_velocity": obs[185:269],
    "actuator": obs[269:286],
    "contact": obs[286:370],
    # Environment-specific
}

for name, part in obs_parts.items():
    print(f"{name}: min={part.min():.2f}, max={part.max():.2f}, mean={part.mean():.2f}")
```

---

## Future Enhancements

Potential observation space extensions:

1. **Previous action:** Add last action to observation (memory)
2. **Step counter:** Normalized episode progress
3. **Velocity grid:** Not just heights, but also surface properties
4. **Extended height grid:** 7×7 or larger for longer-range perception
5. **Egocentric velocity:** Transform velocities to body frame
6. **Contact indicators:** Binary flags for which body parts are in contact

---

## References

- **Gymnasium Humanoid-v5:** [Official Documentation](https://gymnasium.farama.org/environments/mujoco/humanoid/)
- **MuJoCo:** [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- **Observation Design Best Practices:** See `envs/README.md` for task-specific details
