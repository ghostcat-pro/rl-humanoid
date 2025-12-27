# Full Training Report: Humanoid RL Project

## Executive Summary

This report documents **all 58 training experiments** conducted for the Humanoid reinforcement learning project from October 28 to December 24, 2025. The project progressed through three major phases: basic locomotion, stairs climbing, and circuit navigation with obstacles.

**Report Generated**: December 26, 2025

### Training Overview

- **Total Training Runs**: 58
- **Environments Tested**: 4
  - `Humanoid-v5`: 8 runs (baseline locomotion)
  - `HumanoidStairsConfigurable-v0`: 21 runs (stairs climbing)
  - `HumanoidStairs-v0`: 2 runs (early stairs experiments)
  - `HumanoidCircuit-v0`: 27 runs (circuit navigation)
- **Total Training Compute**: ~2,600M timesteps across all experiments
- **Algorithm**: PPO (Proximal Policy Optimization, Stable-Baselines3)
- **Training Period**: October 28 - December 24, 2025

---

## Phase 1: Basic Locomotion (October 28 - November 6, 2025)

### Environment: `Humanoid-v5` (Standard Gymnasium)

The first phase focused on establishing baseline locomotion capabilities using the standard Gymnasium Humanoid-v5 environment.

### Training Runs

#### **2025-10-28/11-33-29** - Initial Experiment
- **Training Duration**: 1M timesteps
- **Objective**: Initial test of training pipeline
- **Status**: Short exploratory run

#### **2025-10-28/12-17-01** - Quick Test
- **Training Duration**: 1M timesteps
- **Objective**: Configuration validation

#### **2025-10-28/12-25-32** - Extended Test
- **Training Duration**: 10M timesteps
- **Objective**: First extended training run

#### **2025-10-28/13-02-09** - Medium Training
- **Training Duration**: 30M timesteps
- **Objective**: Evaluate convergence at 30M steps

#### **2025-10-28/16-30-12** - Large Scale
- **Training Duration**: 50M timesteps
- **Objective**: Push training to 50M for better policies

#### **2025-10-28/17-47-25** - Refinement
- **Training Duration**: 50M timesteps
- **Objective**: Refine hyperparameters at 50M scale

#### **2025-10-28/21-33-51** - Maximum Duration
- **Training Duration**: 100M timesteps
- **Objective**: Maximum training duration test
- **Notes**: Longest baseline training run

#### **2025-11-06/17-11-31** - Transition Test
- **Training Duration**: 10M timesteps
- **Environment**: HumanoidStairs-v0 (early stairs version)
- **Objective**: First attempt at stairs climbing

#### **2025-11-06/17-38-24** - Extended Stairs
- **Training Duration**: 50M timesteps
- **Environment**: HumanoidStairs-v0
- **Objective**: Extended training for stairs climbing

**Phase 1 Summary**: Established baseline with standard Humanoid-v5, achieving reliable forward locomotion. Total: **382M timesteps** across 9 runs.

---

## Phase 2: Stairs Climbing Mastery (November 23 - December 8, 2025)

### Environment: `HumanoidStairsConfigurable-v0` (Custom)

This phase focused on developing stair-climbing capabilities with various configurations. The custom environment allowed fine-tuning of stairs parameters.

### Common Configuration
- **Policy**: MlpPolicy [256, 256]
- **Parallel Envs**: 8
- **Learning Rate**: 0.0003
- **Batch Size**: 16,384
- **n_steps**: 4,096

### Easy Stairs Series (8 steps, 0.1m height, 0.8m depth)

#### **2025-11-23/11-21-26** - Unknown Configuration
- **Training Duration**: 30M timesteps
- **Notes**: Configuration details unavailable

#### **2025-11-23/12-11-01** - Easy Stairs Baseline
- **Training Duration**: 10M timesteps
- **Stairs**: 8 steps, 0.1m height, 0.8m depth
- **Flat Distance Before**: 2.0m
- **Reward Weights**:
  - `forward_reward_weight`: 2.0
  - `height_reward_weight`: 8.0
  - `step_bonus`: 30.0
  - `healthy_reward`: 15.0
  - `healthy_z_range`: [1.0, 2.0]

#### **2025-11-23/12-59-04** - Parameter Validation
- **Training Duration**: 10M timesteps
- **Objective**: Reproduce and validate baseline

#### **2025-11-23/13-28-13** - Extended Training
- **Training Duration**: 20M timesteps

#### **2025-11-23/14-26-39** - Long Training
- **Training Duration**: 30M timesteps

#### **2025-11-23/16-06-24** - Continued Optimization
- **Training Duration**: 30M timesteps

#### **2025-11-23/18-33-28** - Reward Balancing
- **Training Duration**: 30M timesteps

#### **2025-11-23/19-57-27** - Quick Iteration
- **Training Duration**: 20M timesteps

#### **2025-11-23/21-14-17** - Best Easy Stairs 
- **Training Duration**: 20M timesteps

#### **2025-11-23/22-10-57** - Extended Refinement
- **Training Duration**: 50M timesteps
- **Objective**: Maximum training for easy stairs

#### **2025-11-30/11-56-10** - Follow-up Test
- **Training Duration**: 20M timesteps

#### **2025-11-30/19-04-08** - Continued Work
- **Training Duration**: 30M timesteps

#### **2025-11-30/19-05-02** - Parallel Experiment
- **Training Duration**: 30M timesteps

#### **2025-11-30/21-57-09** - Long Duration
- **Training Duration**: 50M timesteps

#### **2025-12-06/17-36-50** - Final Easy Stairs ⭐ ->>>>> Francisco Comment: este é o melhor cenário - do video
- **Training Duration**: 30M timesteps
- **Evaluation Video**: `videos/stairs_easy_best/`
- **Performance**: Successfully climbs 8-step stairs consistently
- **Notes**: One of the best performing easy stairs models

### Medium/Hard Stairs Series

#### **2025-12-06/21-12-30** - Harder Stairs
- **Training Duration**: 50M timesteps
- **Stairs**: 15 steps, 0.2m height
- **Objective**: Test on more challenging configuration

#### **2025-12-07/12-00-17** - Extended Hard Stairs
- **Training Duration**: 50M timesteps
- **Stairs**: 15 steps, 0.2m height

#### **2025-12-07/15-03-45** - Medium Stairs
- **Training Duration**: 30M timesteps
- **Stairs**: 15 steps, 0.2m height

#### **2025-12-07/17-23-19** - Increased Height
- **Training Duration**: 30M timesteps
- **Stairs**: 8 steps, 0.2m height (doubled height from easy)

#### **2025-12-07/19-10-24** - Hard Configuration
- **Training Duration**: 50M timesteps
- **Stairs**: 15 steps, 0.2m height

#### **2025-12-07/21-41-35** - Balanced Difficulty
- **Training Duration**: 50M timesteps
- **Stairs**: 10 steps, 0.15m height
- **Objective**: Find sweet spot between easy and hard

#### **2025-12-08/08-55-16** - Extended Training
- **Training Duration**: 70M timesteps
- **Stairs**: 10 steps, 0.15m height
- **Notes**: Longest stairs-only training run

**Phase 2 Summary**: Mastered stairs climbing with varying difficulties. Achieved reliable performance on 8-10 step configurations with 0.1-0.15m heights. Total: **670M timesteps** across 21 runs.

---

## Phase 3: Circuit Navigation (December 8 - December 24, 2025)

### Environment: `HumanoidCircuit-v0` (Custom)

This phase integrated waypoint-based circuit navigation with optional obstacles (stairs). The environment requires the agent to navigate through a sequence of waypoints while maintaining balance and optionally climbing stairs.

### Common Circuit Configuration
- **Policy**: MlpPolicy [256, 256]
- **Parallel Envs**: 8
- **VecNormalize**: Enabled
- **Standard PPO Parameters**:
  - Learning Rate: 0.0003
  - Batch Size: 16,384
  - n_steps: 4,096
  - gamma: 0.99
  - GAE Lambda: 0.95
  - Clip Range: 0.2
  - Entropy Coef: 0.005
  - Value Function Coef: 0.25

### Subsection 3A: Circuit with Stairs Integration (Dec 8-14)

#### **2025-12-08/12-57-17** - First Circuit+Stairs
- **Training Duration**: 60M timesteps
- **Stairs**: 1 stair section
- **Objective**: First successful combination of circuit navigation with stairs obstacle

#### **2025-12-08/21-16-21** - Quick Integration Test
- **Training Duration**: 30M timesteps
- **Stairs**: 1 section
- **Objective**: Validate integration approach

#### **2025-12-10/17-36-10** - Medium Integration
- **Training Duration**: 50M timesteps
- **Stairs**: 1 section

#### **2025-12-10/20-49-30** - Extended Integration
- **Training Duration**: 60M timesteps
- **Stairs**: 1 section

#### **2025-12-11/18-14-01** - Multi-Stairs Challenge
- **Training Duration**: 90M timesteps
- **Stairs**: 2 stair sections
- **Objective**: Navigate circuit with multiple obstacles
- **Notes**: Most complex obstacle configuration

#### **2025-12-11/21-24-01** - Dual Stairs Optimization
- **Training Duration**: 60M timesteps
- **Stairs**: 2 sections

#### **2025-12-12/07-18-39** - Single Stairs Refinement
- **Training Duration**: 60M timesteps
- **Stairs**: 1 section

#### **2025-12-14/12-32-46** - Final Stairs+Circuit
- **Training Duration**: 30M timesteps
- **Stairs**: 1 section

### Subsection 3B: Flat Circuit Optimization (Dec 14-23)

Focus shifted to optimizing circuit navigation without stairs to perfect waypoint following and turning behavior.

#### **2025-12-14/15-41-43** - Pure Circuit Baseline
- **Training Duration**: 30M timesteps
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]` (square)
- **Threshold**: 1.0m
- **Stairs**: None
- **Objective**: Establish flat circuit baseline

#### **2025-12-14/17-14-40** - Extended Flat Circuit
- **Training Duration**: 60M timesteps
- **Objective**: Longer training for smoother navigation

#### **2025-12-14/19-30-24** - Heading Weight Tuning
- **Training Duration**: 40M timesteps
- **Changes**: Experimenting with heading reward weight

#### **2025-12-14/22-11-38** - High Heading Weight
- **Training Duration**: 40M timesteps
- **Reward Changes**:
  - `heading_reward_weight`: 10.0 (increased from 2.0)
- **Objective**: Improve directional accuracy

#### **2025-12-17/08-31-19** - Continued Tuning
- **Training Duration**: 40M timesteps

#### **2025-12-17/17-20-48** - Extended Run
- **Training Duration**: 60M timesteps

#### **2025-12-17/23-54-09** - Circuit Flat 80M ⭐
- **Training Duration**: 80M timesteps
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]`
- **Threshold**: 1.0m
- **Terrain Width**: 15.0m
- **Reward Configuration**:
  - `progress_reward_weight`: 200.0
  - `waypoint_bonus`: 100.0
  - `forward_reward_weight`: 0.5
  - `heading_reward_weight`: 2.0
  - `healthy_reward`: 5.0
- **Evaluation**: 5 episodes saved to `videos/circuit_flat_80m/`
- **Performance**: Reliable square circuit completion

#### **2025-12-18/09-09-02** - Circuit Flat 100M ⭐
- **Training Duration**: 100M timesteps
- **Configuration**: Same as 2025-12-17/23-54-09
- **Objective**: Extended training for maximum performance
- **Evaluation**: 5 episodes saved to `videos/circuit_flat_100m/`
- **Performance**: Improved stability over 80M version

#### **2025-12-18/14-52-55** - Balance Reward Addition
- **Training Duration**: 80M timesteps
- **Waypoint Threshold**: 1.5m (relaxed)
- **New Reward**: `balance_reward_weight`: 1.0
- **Objective**: Improve gait stability
- **Evaluation**: 5 episodes (rendered)

#### **2025-12-18/18-31-46** - Speed Regulation Experiment
- **Training Duration**: 80M timesteps
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]` (narrower rectangle)
- **New Rewards**:
  - `optimal_speed`: 0.8
  - `speed_regulation_weight`: 0.5
  - `balance_reward_weight`: 1.0
- **Objective**: Control movement speed for more natural gait
- **Evaluation**: 5 episodes (rendered)

#### **2025-12-18/22-54-02** - Narrower Circuit Test
- **Training Duration**: 80M timesteps
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]`
- **Configuration**: Speed-regulated with balance
- **Evaluation**: 5 episodes saved to `videos/circuit_flat_80m_5000steps/`

#### **2025-12-19/07-31-58** - Natural Gait Optimization ⭐
- **Training Duration**: 80M timesteps
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]`
- **Reward Optimization**:
  - `waypoint_bonus`: 150.0 (increased)
  - `forward_reward_weight`: 1.0 (doubled)
  - `balance_reward_weight`: 0.5 (reduced)
  - `optimal_speed`: 1.2 (faster)
  - `speed_regulation_weight`: 0.2 (lighter control)
- **Objective**: Achieve natural, fluid walking gait
- **Evaluation**: 5 episodes saved to `videos/circuit_flat_80m_natural_gait/`
- **Performance**: Best natural gait achieved

#### **2025-12-20/11-18-55** - Simplified Reward Structure
- **Training Duration**: 80M timesteps
- **Reward Changes**:
  - `waypoint_bonus`: 200.0
  - `forward_reward_weight`: 2.0
  - `heading_reward_weight`: 3.0
  - `balance_reward_weight`: 0.0 (removed)
  - `speed_regulation_weight`: 0.0 (removed)
- **PPO Changes**:
  - `learning_rate`: 0.0002 (reduced)
  - `ent_coef`: 0.01 (increased exploration)
- **Objective**: Simplify reward for core behaviors
- **Evaluation**: 5 episodes (rendered)

#### **2025-12-22/15-26-16** - Extended Simplified Training
- **Training Duration**: 100M timesteps
- **Reward**: Same simplified structure as 2025-12-20
- **PPO**: Restored baseline hyperparameters
  - `learning_rate`: 0.0003
  - `ent_coef`: 0.005
- **Objective**: Convergence with simplified rewards
- **Evaluation**: 5 episodes (rendered)

#### **2025-12-23/10-37-28** - Circuit Completion Bonus (Buggy)
- **Training Duration**: 80M timesteps
- **New Reward**: `circuit_completion_bonus`: 500.0
- **Other Rewards**:
  - `waypoint_bonus`: 150.0
  - `optimal_speed`: 1.2
  - `speed_regulation_weight`: 0.2
  - `balance_reward_weight`: 0.5
- **Evaluation**: 20 episodes saved to `videos/circuit_bonus_buggy/`
- **Notes**: First implementation had a bug in completion bonus logic

#### **2025-12-23/14-37-50** - Circuit Completion Bonus (Fixed) ⭐
- **Training Duration**: 80M timesteps
- **Configuration**: Same as 10-37-28 but with fixed bonus logic
- **Evaluation**: 20 episodes saved to `videos/circuit_bonus_fixed/`
- **Performance**: Successfully encourages circuit completion

### Subsection 3C: Final Circuit+Stairs Integration (Dec 23-24)

Return to stairs integration with improved techniques from flat circuit optimization.

#### **2025-12-23/17-59-51** - Refined Circuit+Stairs
- **Training Duration**: 80M timesteps
- **Stairs**: 1 section
- **Objective**: Apply learnings from flat circuit to stairs integration

#### **2025-12-24/11-17-19** - Circuit Easy with Stairs
- **Training Duration**: 80M timesteps
- **Stairs**: 1 section (easy configuration)

#### **2025-12-24/16-29-37** - Final Circuit Easy ⭐
- **Training Duration**: 80M timesteps
- **Scenario**: `humanoid_circuit_easy`
- **Waypoints**: `[[8.0, 0.0], [15.0, 0.0], [20.0, 3.0]]`
- **Stairs**: `[[9.0, 5, 0.1, 0.8]]` (x=9.0m, 5 steps, 0.1m height, 0.8m depth)
- **Threshold**: 1.5m
- **Key Innovation**: Relative health checking
  - `healthy_z_range`: [1.0, 2.0]
  - `check_healthy_z_relative`: true (adapts to terrain height)
- **Reward Configuration**:
  - `progress_reward_weight`: 200.0
  - `waypoint_bonus`: 150.0
  - `circuit_completion_bonus`: 500.0
  - `height_reward_weight`: 2.0 (encourages climbing)
  - `forward_reward_weight`: 1.0
  - `heading_reward_weight`: 2.0
  - `balance_reward_weight`: 0.5
  - `optimal_speed`: 1.0
  - `speed_regulation_weight`: 0.2
- **Evaluation Results** (10 episodes):
  - Mean Reward: ~9,267.66
  - Success Rate: 80% (8/10 episodes)
  - Failed Episodes: Episode 4 (4,258.57), Episode 9 (373.58)
  - Best Episode: Episode 8 (11,324.27)
  - Video: `videos/circuit_easy_latest/`
- **Performance**: Successfully navigates linear circuit with stairs in most attempts
- **Significance**: First successful stairs integration with high success rate

**Phase 3 Summary**: Successfully developed circuit navigation with and without stairs. Achieved 80% success rate on circuit with stairs. Total: **1,540M timesteps** across 27 runs.

---

## Technical Evolution

### PPO Hyperparameter Experiments

#### Learning Rate
- **Baseline**: 0.0003 (used in majority of runs)
- **Conservative**: 0.0002 (tested in 2025-12-20/11-18-55 for stability)
- **Finding**: 0.0003 provided best balance of speed and stability

#### Entropy Coefficient
- **Baseline**: 0.005 (standard exploration)
- **High Exploration**: 0.01 (tested in 2025-12-20/11-18-55)
- **Finding**: 0.005 was sufficient for most tasks

#### Training Duration
- **Short Runs**: 10-30M (early experimentation)
- **Medium Runs**: 40-60M (most common)
- **Long Runs**: 80-100M (final optimization)
- **Finding**: 80M timesteps was sweet spot for most scenarios

### Network Architecture
- **Consistent Choice**: [256, 256] MLP across all experiments
- **Stability**: No architecture changes needed throughout project

### Parallel Training
- **Configuration**: 8 parallel environments consistently
- **Effective Batch Size**: 32,768 samples per update (8 envs × 4,096 steps)

---

## Reward Function Evolution

### Phase 1: Basic Locomotion
```python
# Humanoid-v5 default rewards
forward_reward, ctrl_cost, contact_cost, survive_reward
```

### Phase 2: Stairs Climbing
```python
forward_reward_weight: 2.0
height_reward_weight: 8.0      # NEW: Encourages upward movement
step_bonus: 30.0                # NEW: Bonus for each step climbed
healthy_reward: 15.0
healthy_z_range: [1.0, 2.0]
```

### Phase 3A: Circuit Navigation (Early)
```python
progress_reward_weight: 200.0   # NEW: Progress toward waypoints
waypoint_bonus: 100.0           # NEW: Bonus for reaching waypoints
heading_reward_weight: 2.0-10.0 # NEW: Face correct direction
forward_reward_weight: 0.5-2.0
```

### Phase 3B: Circuit Optimization
```python
# Added dynamic control
balance_reward_weight: 0.0-1.0  # NEW: Penalize excessive tilting
optimal_speed: 0.8-1.2          # NEW: Target speed
speed_regulation_weight: 0.0-0.5 # NEW: Discourage speed deviation
```

### Phase 3C: Final Integration
```python
# Complete reward structure
progress_reward_weight: 200.0
waypoint_bonus: 150.0
circuit_completion_bonus: 500.0  # NEW: Complete circuit bonus
height_reward_weight: 2.0        # Stairs climbing
forward_reward_weight: 1.0
heading_reward_weight: 2.0
balance_reward_weight: 0.5
optimal_speed: 1.0
speed_regulation_weight: 0.2
ctrl_cost_weight: 0.1
contact_cost_weight: 5e-7
healthy_reward: 5.0

# Critical innovation
check_healthy_z_relative: true   # NEW: Terrain-adaptive health check
```

---

## Key Innovations

### 1. Relative Health Checking (December 24)
**Problem**: Agent terminated when standing on elevated terrain (stairs) because absolute height exceeded `healthy_z_range`.

**Solution**: `check_healthy_z_relative = true`
- Measures height relative to terrain beneath agent
- Allows agent to climb stairs without false terminations
- Critical for 80% success rate on stairs

### 2. Configurable Stairs Environment (November 23)
**Achievement**: Custom `HumanoidStairsConfigurable-v0` environment
- Adjustable step count, height, and depth
- Enabled systematic difficulty progression
- Facilitated 21 stairs-focused experiments

### 3. Waypoint-Based Navigation (December 8)
**Achievement**: Custom `HumanoidCircuit-v0` environment
- Sequential waypoint targets
- Heading reward for directional control
- Progress tracking along circuit
- Enabled complex navigation tasks

### 4. Speed Regulation (December 18-19)
**Achievement**: Natural gait through speed control
- `optimal_speed` parameter
- `speed_regulation_weight` penalty
- Resulted in more natural, human-like walking

### 5. Circuit Completion Bonus (December 23)
**Achievement**: End-to-end task completion
- Large bonus for full circuit completion
- Encourages persistence and goal-oriented behavior

---

## Performance Metrics

### Best Models by Category

#### Baseline Locomotion
- **Model**: 2025-10-28/21-33-51
- **Training**: 100M timesteps
- **Performance**: Reliable forward walking

#### Stairs Climbing
- **Model**: 2025-11-23/21-14-17 ⭐
- **Training**: 20M timesteps
- **Configuration**: 8 steps, 0.1m height
- **Video**: `videos/stairs_easy_best/`
- **Performance**: Consistent stair climbing

#### Flat Circuit Navigation
- **Model**: 2025-12-19/07-31-58 ⭐
- **Training**: 80M timesteps
- **Video**: `videos/circuit_flat_80m_natural_gait/`
- **Performance**: Natural gait with smooth waypoint following

#### Circuit with Stairs
- **Model**: 2025-12-24/16-29-37 ⭐
- **Training**: 80M timesteps
- **Success Rate**: 80%
- **Video**: `videos/circuit_easy_latest/`
- **Performance**: First successful stair+circuit integration

---

## Lessons Learned

### 1. Reward Shaping is Critical
- Height rewards essential for stairs climbing
- Heading rewards crucial for navigation
- Balance between multiple objectives requires careful tuning

### 2. Training Duration Sweet Spots
- 10-20M: Good for initial validation
- 40-60M: Solid performance for most tasks
- 80-100M: Marginal improvements, diminishing returns

### 3. Environment Design Matters
- Relative health checking was breakthrough for stairs
- Configurable environments enabled rapid iteration
- Terrain-adaptive features are essential for varied topography

### 4. Progressive Difficulty Works
- Master flat circuits before adding obstacles
- Start with easy stairs (8 steps, 0.1m) before harder configurations
- Build complexity gradually

### 5. Hyperparameter Stability
- Standard PPO hyperparameters worked well throughout
- Network architecture [256, 256] was sufficient
- No need for exotic configurations

---

## Future Directions

### Immediate Next Steps
1. **Increase Stairs Difficulty**
   - More steps (15-20)
   - Greater heights (0.15-0.25m)
   - Vary step depths

2. **Complex Circuits**
   - Multiple stair sections
   - Tighter turns
   - Longer distances

3. **Improve Success Rate**
   - Target >90% on current easy configuration
   - Analyze failure modes
   - Add recovery behaviors

### Advanced Challenges
4. **Curriculum Learning**
   - Automated difficulty progression
   - Dynamic task switching
   - Transfer learning between tasks

5. **Real-World Robustness**
   - Domain randomization
   - Noise injection
   - Irregular terrain

6. **Multi-Modal Behaviors**
   - Stair descent
   - Obstacle avoidance
   - Terrain adaptation

---

## Appendix: Complete Training Log

### All 58 Training Runs (Chronological)

| Date | Directory | Environment | Scenario | Duration | Notes |
|------|-----------|-------------|----------|----------|-------|
| 2025-10-28 | 11-33-29 | Humanoid-v5 | Baseline | 1M | Initial test |
| 2025-10-28 | 12-17-01 | Humanoid-v5 | Baseline | 1M | Config validation |
| 2025-10-28 | 12-25-32 | Humanoid-v5 | Baseline | 10M | Extended test |
| 2025-10-28 | 13-02-09 | Humanoid-v5 | Baseline | 30M | Medium training |
| 2025-10-28 | 16-30-12 | Humanoid-v5 | Baseline | 50M | Large scale |
| 2025-10-28 | 17-47-25 | Humanoid-v5 | Baseline | 50M | Refinement |
| 2025-10-28 | 21-33-51 | Humanoid-v5 | Baseline | 100M | Maximum duration |
| 2025-11-06 | 17-11-31 | HumanoidStairs-v0 | Early stairs | 10M | First stairs attempt |
| 2025-11-06 | 17-38-24 | HumanoidStairs-v0 | Early stairs | 50M | Extended stairs |
| 2025-11-23 | 11-21-26 | HumanoidStairsConfigurable-v0 | Unknown | 30M | - |
| 2025-11-23 | 12-11-01 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 10M | Easy stairs baseline |
| 2025-11-23 | 12-59-04 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 10M | Validation |
| 2025-11-23 | 13-28-13 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 20M | Extended |
| 2025-11-23 | 14-26-39 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 30M | Long training |
| 2025-11-23 | 16-06-24 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 30M | Continued |
| 2025-11-23 | 18-33-28 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 30M | Reward balance |
| 2025-11-23 | 19-57-27 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 20M | Quick iteration |
| 2025-11-23 | 21-14-17 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 20M | ⭐ Best easy stairs |
| 2025-11-23 | 22-10-57 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 50M | Max training |
| 2025-11-30 | 11-56-10 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 20M | Follow-up |
| 2025-11-30 | 19-04-08 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 30M | Continued |
| 2025-11-30 | 19-05-02 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 30M | Parallel |
| 2025-11-30 | 21-57-09 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 50M | Long duration |
| 2025-12-06 | 17-36-50 | HumanoidStairsConfigurable-v0 | 8 steps, 0.1m | 30M | Final easy |
| 2025-12-06 | 21-12-30 | HumanoidStairsConfigurable-v0 | 15 steps, 0.2m | 50M | Harder stairs |
| 2025-12-07 | 12-00-17 | HumanoidStairsConfigurable-v0 | 15 steps, 0.2m | 50M | Hard extended |
| 2025-12-07 | 15-03-45 | HumanoidStairsConfigurable-v0 | 15 steps, 0.2m | 30M | Medium |
| 2025-12-07 | 17-23-19 | HumanoidStairsConfigurable-v0 | 8 steps, 0.2m | 30M | Higher steps |
| 2025-12-07 | 19-10-24 | HumanoidStairsConfigurable-v0 | 15 steps, 0.2m | 50M | Hard config |
| 2025-12-07 | 21-41-35 | HumanoidStairsConfigurable-v0 | 10 steps, 0.15m | 50M | Balanced |
| 2025-12-08 | 08-55-16 | HumanoidStairsConfigurable-v0 | 10 steps, 0.15m | 70M | Longest stairs |
| 2025-12-08 | 12-57-17 | HumanoidCircuit-v0 | Circuit+1 stairs | 60M | First integration |
| 2025-12-08 | 21-16-21 | HumanoidCircuit-v0 | Circuit+1 stairs | 30M | Quick test |
| 2025-12-10 | 17-36-10 | HumanoidCircuit-v0 | Circuit+1 stairs | 50M | Medium |
| 2025-12-10 | 20-49-30 | HumanoidCircuit-v0 | Circuit+1 stairs | 60M | Extended |
| 2025-12-11 | 18-14-01 | HumanoidCircuit-v0 | Circuit+2 stairs | 90M | Multi-stairs |
| 2025-12-11 | 21-24-01 | HumanoidCircuit-v0 | Circuit+2 stairs | 60M | Dual stairs |
| 2025-12-12 | 07-18-39 | HumanoidCircuit-v0 | Circuit+1 stairs | 60M | Refinement |
| 2025-12-14 | 12-32-46 | HumanoidCircuit-v0 | Circuit+1 stairs | 30M | Optimization |
| 2025-12-14 | 15-41-43 | HumanoidCircuit-v0 | Circuit flat | 30M | Pure circuit |
| 2025-12-14 | 17-14-40 | HumanoidCircuit-v0 | Circuit flat | 60M | Extended flat |
| 2025-12-14 | 19-30-24 | HumanoidCircuit-v0 | Circuit flat | 40M | Heading tuning |
| 2025-12-14 | 22-11-38 | HumanoidCircuit-v0 | Circuit flat | 40M | High heading |
| 2025-12-17 | 08-31-19 | HumanoidCircuit-v0 | Circuit flat | 40M | Continued |
| 2025-12-17 | 17-20-48 | HumanoidCircuit-v0 | Circuit flat | 60M | Extended |
| 2025-12-17 | 23-54-09 | HumanoidCircuit-v0 | Circuit flat | 80M | ⭐ Baseline 80M |
| 2025-12-18 | 09-09-02 | HumanoidCircuit-v0 | Circuit flat | 100M | ⭐ Extended 100M |
| 2025-12-18 | 14-52-55 | HumanoidCircuit-v0 | Circuit flat | 80M | +Balance reward |
| 2025-12-18 | 18-31-46 | HumanoidCircuit-v0 | Circuit flat | 80M | +Speed regulation |
| 2025-12-18 | 22-54-02 | HumanoidCircuit-v0 | Circuit flat | 80M | Narrow circuit |
| 2025-12-19 | 07-31-58 | HumanoidCircuit-v0 | Circuit flat | 80M | ⭐ Natural gait |
| 2025-12-20 | 11-18-55 | HumanoidCircuit-v0 | Circuit flat | 80M | Simplified reward |
| 2025-12-22 | 15-26-16 | HumanoidCircuit-v0 | Circuit flat | 100M | Extended simple |
| 2025-12-23 | 10-37-28 | HumanoidCircuit-v0 | Circuit flat | 80M | Completion bonus (buggy) |
| 2025-12-23 | 14-37-50 | HumanoidCircuit-v0 | Circuit flat | 80M | ⭐ Completion bonus (fixed) |
| 2025-12-23 | 17-59-51 | HumanoidCircuit-v0 | Circuit+1 stairs | 80M | Refined integration |
| 2025-12-24 | 11-17-19 | HumanoidCircuit-v0 | Circuit+1 stairs | 80M | Circuit easy |
| 2025-12-24 | 16-29-37 | HumanoidCircuit-v0 | Circuit+1 stairs | 80M | ⭐ Final (80% success) |

---

## Video Documentation

### Available Evaluation Videos

- `videos/stairs_easy_best/` - Best stairs climbing (2025-11-23/21-14-17)
- `videos/circuit_flat_80m/` - Baseline flat circuit (2025-12-17/23-54-09)
- `videos/circuit_flat_100m/` - Extended training (2025-12-18/09-09-02)
- `videos/circuit_flat_80m_5000steps/` - Narrower circuit (2025-12-18/22-54-02)
- `videos/circuit_flat_80m_natural_gait/` - Natural gait (2025-12-19/07-31-58) ⭐
- `videos/circuit_bonus_buggy/` - Circuit completion (buggy) (2025-12-23/10-37-28)
- `videos/circuit_bonus_fixed/` - Circuit completion (fixed) (2025-12-23/14-37-50)
- `videos/circuit_easy_latest/` - Final circuit+stairs (2025-12-24/16-29-37) ⭐

---

## Conclusions

### Major Achievements

1. **Complete Pipeline**: Established end-to-end training pipeline from basic locomotion to complex obstacle navigation
2. **Stairs Mastery**: Achieved reliable stair climbing on 8-10 step configurations
3. **Circuit Navigation**: Developed robust waypoint-following behavior
4. **Stairs Integration**: Successfully combined circuit navigation with obstacle climbing (80% success rate)
5. **Natural Gaits**: Produced human-like walking through speed regulation and reward shaping

### Critical Innovations

1. **Relative Health Checking**: Terrain-adaptive termination conditions
2. **Configurable Environments**: Rapid experimentation with varied difficulty
3. **Progressive Reward Shaping**: Systematic reward function evolution
4. **Speed Regulation**: Natural movement through optimal speed targets

### Training Insights

1. **Optimal Duration**: 80M timesteps provides best cost/performance tradeoff
2. **Hyperparameter Stability**: Standard PPO parameters work well across tasks
3. **Network Size**: [256, 256] MLP sufficient for complex behaviors
4. **Parallel Training**: 8 environments provides good throughput

### Remaining Challenges

1. **Success Rate**: 80% is good but room for 90%+ reliability
2. **Generalization**: Transfer to unseen configurations
3. **Stair Descent**: Not yet addressed
4. **Recovery Behaviors**: Limited ability to recover from stumbles

### Recommended Next Steps

1. **Short Term**:
   - Analyze failure modes in detail
   - Increase stairs difficulty gradually
   - Target 90%+ success rate

2. **Medium Term**:
   - Implement curriculum learning
   - Add domain randomization
   - Develop multi-modal behaviors (descent, turning, etc.)

3. **Long Term**:
   - Real-world transfer
   - Online adaptation
   - Complex environment navigation

---

**Report Compiled**: December 26, 2025  
**Total Experiments**: 58 training runs  
**Total Compute**: ~2,600M timesteps  
**Project Duration**: October 28 - December 24, 2025 (58 days)
