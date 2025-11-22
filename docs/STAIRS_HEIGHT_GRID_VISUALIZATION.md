# Stairs Height Grid Visualization

## Stair Geometry

The stairs environment has the following structure:

```
                                    ┌─────────────┐ End Platform (h=1.5m)
                               ┌────┤ Step 10     │ h=1.35m
                          ┌────┤    └─────────────┘
                     ┌────┤ Step 9                  h=1.20m
                ┌────┤ Step 8                       h=1.05m
           ┌────┤ Step 7                            h=0.90m
      ┌────┤ Step 6                                 h=0.75m
 ┌────┤ Step 5                                      h=0.60m
─┤ Step 4                                           h=0.45m
 │ Step 3                                           h=0.30m
 │ Step 2                                           h=0.15m
 │ Step 1                                           h=0.00m
─┴─────────┘
Start Platform (h=0)

x: 0    1.0  1.6  2.2  2.8  3.4  4.0  4.6  5.2  5.8  6.4  7.0
```

### Dimensions:
- **Start platform**: x < 1.0m, height = 0m
- **Each step**: 
  - Length: 0.6m (in x direction)
  - Height: 0.15m
  - 10 total steps
- **End platform**: x ≥ 7.0m, height = 1.5m

---

## Height Grid Implementation

The agent has a **5×5 height grid** that samples terrain around it:

### Grid Parameters:
- **Size**: 5×5 = 25 sample points
- **Spacing**: 0.3m between samples
- **Coverage**: 1.2m × 1.2m square centered on agent
- **Values**: Relative height (terrain_height - agent_current_z)

### Grid Layout (top view):

```
        Y
        ↑
        │
    -0.6│  • • • • •
        │  
    -0.3│  • • • • •
        │  
     0.0│  • • A • •  ← Agent at center
        │  
    +0.3│  • • • • •
        │  
    +0.6│  • • • • •
        │
        └────────────────→ X
           -0.6  -0.3  0  +0.3  +0.6

Legend:
• = Sample point
A = Agent position (center of grid)
```

---

## Example Scenarios

### Scenario 1: Agent on Start Platform (x=0.5)

**Agent Position**: x=0.5, z=1.4 (standing height)

**Grid samples** (x positions):
```
Row 0 (back):   -0.1,  0.2,  0.5,  0.8,  1.1
Row 1:          -0.1,  0.2,  0.5,  0.8,  1.1
Row 2 (center): -0.1,  0.2,  0.5,  0.8,  1.1  ← Agent at x=0.5
Row 3:          -0.1,  0.2,  0.5,  0.8,  1.1
Row 4 (front):  -0.1,  0.2,  0.5,  0.8,  1.1
```

**Terrain heights at these x positions**:
```
x < 1.0  → height = 0.0m (platform)
x = 1.1  → height = 0.0m (step 1 starts at x=1.0)
```

**Relative heights** (terrain - agent_z = 0.0 - 1.4):
```
All samples ≈ -1.4m (ground is below agent)
Front-right sample (x=1.1) ≈ -1.4m (still on platform)
```

**What agent "sees"**: Flat platform, all samples at same height below.

---

### Scenario 2: Agent Approaching First Step (x=0.8)

**Agent Position**: x=0.8, z=1.4

**Grid samples** (x positions):
```
Row 0 (back):    0.2,  0.5,  0.8,  1.1,  1.4
Row 1:           0.2,  0.5,  0.8,  1.1,  1.4
Row 2 (center):  0.2,  0.5,  0.8,  1.1,  1.4  ← Agent at x=0.8
Row 3:           0.2,  0.5,  0.8,  1.1,  1.4
Row 4 (front):   0.2,  0.5,  0.8,  1.1,  1.4
```

**Terrain heights**:
```
x = 0.2, 0.5, 0.8  → 0.0m (platform)
x = 1.1, 1.4       → 0.0m (step 1, height index 0)
```

**Relative heights**:
```
Back 3 columns:  -1.4m (platform)
Front 2 columns: -1.4m (step 1 at same level as platform)
```

**What agent "sees"**: Still flat, but first step edge is detectable if we're closer.

---

### Scenario 3: Agent ON First Step (x=1.3)

**Agent Position**: x=1.3, z=1.55 (standing on step 1, height 0.15m + body height)

**Grid samples** (x positions):
```
Row 0 (back):    0.7,  1.0,  1.3,  1.6,  1.9
Row 1:           0.7,  1.0,  1.3,  1.6,  1.9
Row 2 (center):  0.7,  1.0,  1.3,  1.6,  1.9  ← Agent at x=1.3
Row 3:           0.7,  1.0,  1.3,  1.6,  1.9
Row 4 (front):   0.7,  1.0,  1.3,  1.6,  1.9
```

**Terrain heights**:
```
x = 0.7  → 0.00m (platform)
x = 1.0  → 0.00m (edge of platform/step 1 boundary)
x = 1.3  → 0.15m (step 1)
x = 1.6  → 0.15m (step 1)  [step 1 ends at x=1.6]
x = 1.9  → 0.15m (step 2)
```

**Relative heights** (terrain - 1.55):
```
Back-left (x=0.7):   0.0 - 1.55 = -1.55m (platform behind, lower)
Center (x=1.3):      0.15 - 1.55 = -1.40m (current step)
Front-right (x=1.9): 0.15 - 1.55 = -1.40m (next step at same height)
```

**What agent "sees"**: Platform behind is lower, current and next step at same level.

---

### Scenario 4: Agent BETWEEN Steps 2 and 3 (x=2.5)

**Agent Position**: x=2.5, z=1.70 (stepping up, between step 2 at 0.30m and step 3 at 0.45m)

**Grid samples** (x positions):
```
Row 2 (center):  1.9,  2.2,  2.5,  2.8,  3.1
```

**Terrain heights**:
```
x = 1.9  → 0.15m (step 2)
x = 2.2  → 0.30m (step 3, starts at x=2.2)
x = 2.5  → 0.30m (step 3)
x = 2.8  → 0.45m (step 4, starts at x=2.8)
x = 3.1  → 0.45m (step 4)
```

**Relative heights** (terrain - 1.70):
```
Back-left (x=1.9):   0.15 - 1.70 = -1.55m
Back-center (x=2.2): 0.30 - 1.70 = -1.40m
Center (x=2.5):      0.30 - 1.70 = -1.40m  ← Agent here
Front-center (x=2.8): 0.45 - 1.70 = -1.25m ← STEP UP AHEAD!
Front-right (x=3.1):  0.45 - 1.70 = -1.25m
```

**What agent "sees"**: 
- Behind: lower terrain (-1.55m)
- Current: medium terrain (-1.40m)
- **Ahead: HIGHER terrain (-1.25m)** ← Step detected!

This is the **key signal** for learning to step up!

---

### Scenario 5: Agent Near Top (x=6.5)

**Agent Position**: x=6.5, z=2.85 (on step 10, height 1.35m + body height)

**Grid samples** (x positions):
```
Row 2 (center):  5.9,  6.2,  6.5,  6.8,  7.1
```

**Terrain heights**:
```
x = 5.9  → 1.20m (step 9)
x = 6.2  → 1.20m (step 9)
x = 6.5  → 1.35m (step 10)
x = 6.8  → 1.35m (step 10)
x = 7.1  → 1.50m (end platform!)
```

**Relative heights** (terrain - 2.85):
```
Back (x=5.9, 6.2):    1.20 - 2.85 = -1.65m (step behind, lower)
Center (x=6.5, 6.8):  1.35 - 2.85 = -1.50m (current step)
Front (x=7.1):        1.50 - 2.85 = -1.35m (end platform, higher!)
```

**What agent "sees"**: Approaching the top platform!

---

## Key Insights

### What the Height Grid Enables:

1. **Step Detection**: Agent can "see" upcoming steps as terrain gets higher ahead
2. **Edge Awareness**: Can detect when platform/step ends
3. **Planning**: Knows terrain 0.6m ahead (2 steps forward in grid)
4. **Balance**: Can sense if terrain is uneven left/right
5. **Goal Recognition**: Can detect reaching flat top platform

### Learning Signals:

The **relative height gradient** in the forward direction tells the agent:
- Negative gradient (ahead is lower) → Don't go there / falling
- Zero gradient (flat) → Safe to walk normally
- **Positive gradient (ahead is higher)** → STEP UP needed!

### Example Height Grid Values (agent at x=2.5, on step 3):

```
Grid (5x5, showing relative heights):

     -0.6  -0.3   0   +0.3  +0.6  (meters left/right)
      ↓     ↓     ↓     ↓     ↓
-0.6: -1.55 -1.55 -1.40 -1.40 -1.25
-0.3: -1.55 -1.55 -1.40 -1.40 -1.25
 0.0: -1.55 -1.55 -1.40 -1.25 -1.25  ← Agent at center
+0.3: -1.55 -1.55 -1.40 -1.40 -1.25
+0.6: -1.55 -1.55 -1.40 -1.40 -1.25
       ↑                 ↑
     Behind            Ahead (step up!)
```

The agent learns: "When I see -1.25 ahead and -1.40 at center, I need to lift my leg higher!"

---

## Summary

The 5×5 height grid gives the humanoid:
- ✅ **Forward vision**: 0.6m ahead (2 sample points)
- ✅ **Backward awareness**: 0.6m behind
- ✅ **Lateral perception**: 0.6m left and right
- ✅ **Step prediction**: Can anticipate next 1-2 steps
- ✅ **Terrain comprehension**: Knows relative height changes

This should **significantly improve** learning for stair climbing compared to no terrain perception!
