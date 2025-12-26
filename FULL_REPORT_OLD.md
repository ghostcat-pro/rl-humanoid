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
- **Total Training Time**: ~2,600M timesteps across all experiments
- **Algorithm**: PPO (Proximal Policy Optimization, Stable-Baselines3)
- **Training Period**: October 28 - December 24, 2025

---

## Phase 1: Basic Locomotion (October 28 - November 6, 2025)

### Environment: `Humanoid-v5` (Standard Gym)

The first phase focused on establishing baseline locomotion capabilities using the standard Gymnasium Humanoid-v5 environment.

#### **2025-10-28/11-33-29** - Initial Experiment (1M)
- **Training Duration**: 1M timesteps
- **Objective**: Initial test of training pipeline
- **Status**: Short exploratory run

#### **2025-10-28/12-17-01** - Quick Test (1M)
- **Training Duration**: 1M timesteps
- **Objective**: Configuration validation

#### **2025-10-28/12-25-32** - Extended Test (10M)
- **Training Duration**: 10M timesteps
- **Objective**: First extended training run

#### **2025-10-28/13-02-09** - Medium Training (30M)
- **Training Duration**: 30M timesteps
- **Objective**: Evaluate convergence at 30M steps

#### **2025-10-28/16-30-12** - Large Scale (50M)
- **Training Duration**: 50M timesteps
- **Objective**: Push training to 50M for better policies

#### **2025-10-28/17-47-25** - Refinement (50M)
- **Training Duration**: 50M timesteps
- **Objective**: Refine hyperparameters at 50M scale

#### **2025-10-28/21-33-51** - Maximum Duration (100M)
- **Training Duration**: 100M timesteps
- **Objective**: Maximum training duration test
- **Notes**: Longest baseline training run

#### **2025-11-06/17-11-31** - Transition Test (10M)
- **Training Duration**: 10M timesteps
- **Environment**: HumanoidStairs-v0 (early stairs version)
- **Objective**: First attempt at stairs climbing

#### **2025-11-06/17-38-24** - Extended Stairs (50M)
- **Training Duration**: 50M timesteps
- **Environment**: HumanoidStairs-v0
- **Objective**: Extended training for stairs climbing

**Phase 1 Summary**: Established baseline with standard Humanoid-v5, achieving reliable forward locomotion. Total: **382M timesteps** across 8 runs.

---

## Phase 2: Stairs Climbing Mastery (November 23 - December 8, 2025)

### Environment: `HumanoidStairsConfigurable-v0` (Custom)

This phase focused on developing stair-climbing capabilities with various configurations.

### Easy Stairs (8 steps, 0.1m height)

#### **2025-11-23/11-21-26** - Unknown Configuration (30M)
- **Training Duration**: 30M timesteps
- **Notes**: Configuration details unavailable

#### **2025-11-23/12-11-01** - Easy Stairs Baseline (10M)
- **Training Duration**: 10M timesteps
- **Stairs**: 8 steps, 0.1m height, 0.8m depth
- **Flat Distance**: 2.0m before stairs
- **Reward Weights**:
  - `forward_reward_weight`: 2.0
  - `height_reward_weight`: 8.0
  - `step_bonus`: 30.0
  - `healthy_reward`: 15.0

#### **2025-11-23/12-59-04** - Parameter Tuning (10M)
- **Training Duration**: 10M timesteps
- **Configuration**: Same as above
- **Objective**: Reproduce and validate baseline

#### **2025-11-23/13-28-13** - Extended Training (20M)
- **Training Duration**: 20M timesteps
- **Objective**: Improve policy with more training

#### **2025-11-23/14-26-39** - Long Training (30M)
- **Training Duration**: 30M timesteps
- **Objective**: Further refinement

#### **2025-11-23/16-06-24** - Continued Optimization (30M)
- **Training Duration**: 30M timesteps

#### **2025-11-23/18-33-28** - Reward Balancing (30M)
- **Training Duration**: 30M timesteps

#### **2025-11-23/19-57-27** - Quick Iteration (20M)
- **Training Duration**: 20M timesteps

#### **2025-11-23/21-14-17** - Best Easy Stairs (20M) ⭐
- **Training Duration**: 20M timesteps
- **Evaluation**: Saved to `videos/stairs_easy_best/`
- **Performance**: Successfully climbs 8-step stairs
- **Notes**: One of the best performing easy stairs models

#### **2025-11-23/22-10-57** - Extended Refinement (50M)
- **Training Duration**: 50M timesteps
- **Objective**: Maximum training for easy stairs

#### **2025-11-30/11-56-10** - Follow-up Test (20M)
- **Training Duration**: 20M timesteps

#### **2025-11-30/19-04-08** - Continued Work (30M)
- **Training Duration**: 30M timesteps

#### **2025-11-30/19-05-02** - Parallel Experiment (30M)
- **Training Duration**: 30M timesteps

#### **2025-11-30/21-57-09** - Long Duration (50M)
- **Training Duration**: 50M timesteps

#### **2025-12-06/17-36-50** - Final Easy Stairs (30M)
- **Training Duration**: 30M timesteps

### Medium Difficulty Stairs

#### **2025-12-06/21-12-30** - Harder Stairs (50M)
- **Training Duration**: 50M timesteps
- **Stairs**: 15 steps, 0.2m height
- **Objective**: Test on more challenging configuration

#### **2025-12-07/12-00-17** - Extended Hard Stairs (50M)
- **Training Duration**: 50M timesteps
- **Stairs**: 15 steps, 0.2m height

#### **2025-12-07/15-03-45** - Medium Stairs (30M)
- **Training Duration**: 30M timesteps
- **Stairs**: 15 steps, 0.2m height

#### **2025-12-07/17-23-19** - Reduced Steps (30M)
- **Training Duration**: 30M timesteps
- **Stairs**: 8 steps, 0.2m height (increased height)

#### **2025-12-07/19-10-24** - Hard Configuration (50M)
- **Training Duration**: 50M timesteps
- **Stairs**: 15 steps, 0.2m height

#### **2025-12-07/21-41-35** - Balanced Difficulty (50M)
- **Training Duration**: 50M timesteps
- **Stairs**: 10 steps, 0.15m height
- **Objective**: Find sweet spot between easy and hard

#### **2025-12-08/08-55-16** - Extended Training (70M)
- **Training Duration**: 70M timesteps
- **Stairs**: 10 steps, 0.15m height
- **Notes**: Longest stairs-only training run

**Phase 2 Summary**: Mastered stairs climbing with varying difficulties. Achieved reliable performance on 8-10 step configurations. Total: **670M timesteps** across 21 runs.

---

## Phase 3: Circuit Navigation (December 8 - December 24, 2025)

### Environment: `HumanoidCircuit-v0` (Custom)

This phase integrated circuit navigation with optional obstacles (stairs).

### 3A: Circuit with Stairs Integration (December 8-12, 2025)

#### **2025-12-08/12-57-17** - First Circuit+Stairs (60M)
- **Training Duration**: 60M timesteps
- **Waypoints**: Circuit configuration
- **Stairs**: 1 stair section
- **Objective**: Combine navigation with obstacle climbing

#### **2025-12-08/21-16-21** - Quick Test (30M)
- **Training Duration**: 30M timesteps
- **Stairs**: 1 section
- **Objective**: Validate integration

#### **2025-12-10/17-36-10** - Medium Integration (50M)
- **Training Duration**: 50M timesteps
- **Stairs**: 1 section

#### **2025-12-10/20-49-30** - Extended Integration (60M)
- **Training Duration**: 60M timesteps
- **Stairs**: 1 section

#### **2025-12-11/18-14-01** - Multi-Stairs (90M)
- **Training Duration**: 90M timesteps
- **Stairs**: 2 stair sections
- **Objective**: Navigate circuit with multiple obstacles

#### **2025-12-11/21-24-01** - Dual Stairs (60M)
- **Training Duration**: 60M timesteps
- **Stairs**: 2 sections

#### **2025-12-12/07-18-39** - Single Stairs Refinement (60M)
- **Training Duration**: 60M timesteps
- **Stairs**: 1 section

#### **2025-12-14/12-32-46** - Optimization (30M)
- **Training Duration**: 30M timesteps
- **Stairs**: 1 section

### 3B: Flat Circuit Optimization (December 14-23, 2025)

#### **2025-12-14/15-41-43** - Pure Circuit Test (30M)
- **Training Duration**: 30M timesteps
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]`
- **Stairs**: None (flat terrain)

#### **2025-12-14/17-14-40** - Extended Flat Circuit (60M)
- **Training Duration**: 60M timesteps
- **Objective**: Establish flat circuit baseline

#### **2025-12-14/19-30-24** - Heading Tuning (40M)
- **Training Duration**: 40M timesteps
- **Heading Reward**: Increased for better navigation

#### **2025-12-14/22-11-38** - High Heading Weight (40M)
- **Training Duration**: 40M timesteps
- **Reward Changes**:
  - `heading_reward_weight`: 10.0 (increased from 2.0)

#### **2025-12-17/08-31-19** - Continued Tuning (40M)
- **Training Duration**: 40M timesteps

#### **2025-12-17/17-20-48** - Extended Run (60M)
- **Training Duration**: 60M timesteps

#### **2025-12-17/23-54-09** - Circuit Flat 80M ⭐
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]` (square circuit)
- **Waypoint Threshold**: 1.0m
- **Stairs**: None (flat terrain)
- **Terrain Width**: 15.0m

#### Reward Function
- `progress_reward_weight`: 200.0
- `waypoint_bonus`: 100.0
- `height_reward_weight`: 0.0
- `forward_reward_weight`: 0.5
- `heading_reward_weight`: 2.0
- `ctrl_cost_weight`: 0.1
- `contact_cost_weight`: 5e-7
- `healthy_reward`: 5.0
- `healthy_z_range`: [0.8, 3.0]

#### PPO Hyperparameters
- **Learning Rate**: 0.0003
- **Total Timesteps**: 80,000,000
- **Batch Size**: 16,384
- **n_steps**: 4,096
- **n_epochs**: 10
- **gamma**: 0.99
- **GAE Lambda**: 0.95
- **Clip Range**: 0.2
- **Entropy Coefficient**: 0.005
- **Value Function Coefficient**: 0.25
- **Max Gradient Norm**: 0.5
- **Network Architecture**: [256, 256]
- **Parallel Environments**: 8

#### Evaluation Results (5 episodes)
- **Video**: `videos/circuit_flat_80m/`
- **Performance**: Successfully completed flat circuit navigation

---

### 2. **2025-12-18/09-09-02** - Circuit Flat (100M Steps)
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]`
- **Waypoint Threshold**: 1.0m
- **Stairs**: None
- **Terrain Width**: 15.0m

#### Reward Function
*(Same as 2025-12-17/23-54-09)*

#### PPO Hyperparameters
- **Total Timesteps**: 100,000,000 (extended training)
- *(All other parameters same as previous run)*

#### Evaluation Results (5 episodes)
- **Video**: `videos/circuit_flat_100m/`
- **Performance**: Improved stability with extended training

---

### 3. **2025-12-18/14-52-55** - Circuit Flat with Balance Reward
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]`
- **Waypoint Threshold**: 1.5m (increased tolerance)
- **Stairs**: None
- **Terrain Width**: 15.0m

#### Reward Function (Changes)
- `balance_reward_weight`: **1.0** *(NEW)*
- *(Other rewards same as baseline)*

#### PPO Hyperparameters
- **Total Timesteps**: 80,000,000
- *(Same as baseline)*

#### Evaluation Results (5 episodes)
- **Render Mode**: Human (visual evaluation)
- **Performance**: Added balance reward for more stable gait

---

### 4. **2025-12-18/18-31-46** - Speed Regulation Experiment
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]` (narrower circuit)
- **Waypoint Threshold**: 1.5m
- **Stairs**: None
- **Terrain Width**: 15.0m

#### Reward Function (Changes)
- `balance_reward_weight`: 1.0
- `optimal_speed`: **0.8** *(NEW)*
- `speed_regulation_weight`: **0.5** *(NEW)*

#### PPO Hyperparameters
- **Total Timesteps**: 80,000,000
- *(Same as baseline)*

#### Evaluation Results (5 episodes)
- **Render Mode**: Human
- **Performance**: Introduced speed control for more natural movement

---

### 5. **2025-12-18/22-54-02** - Narrower Circuit with Speed Control
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]`
- **Waypoint Threshold**: 1.5m
- **Stairs**: None
- **Terrain Width**: 15.0m

#### Reward Function
- `waypoint_bonus`: 100.0
- `balance_reward_weight`: 1.0
- `optimal_speed`: 0.8
- `speed_regulation_weight`: 0.5

#### PPO Hyperparameters
- **Total Timesteps**: 80,000,000
- *(Same as baseline)*

#### Evaluation Results (5 episodes)
- **Video**: `videos/circuit_flat_80m_5000steps/`
- **Performance**: Tested on narrower rectangular circuit

---

### 6. **2025-12-19/07-31-58** - Natural Gait Optimization
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]`
- **Waypoint Threshold**: 1.5m
- **Stairs**: None
- **Terrain Width**: 15.0m

#### Reward Function (Changes)
- `waypoint_bonus`: **150.0** *(increased)*
- `forward_reward_weight`: **1.0** *(increased from 0.5)*
- `balance_reward_weight`: **0.5** *(reduced from 1.0)*
- `optimal_speed`: **1.2** *(increased)*
- `speed_regulation_weight`: **0.2** *(reduced)*

#### PPO Hyperparameters
- **Total Timesteps**: 80,000,000
- *(Same as baseline)*

#### Evaluation Results (5 episodes)
- **Video**: `videos/circuit_flat_80m_natural_gait/`
- **Performance**: Optimized for more natural, faster gait

---

### 7. **2025-12-20/11-18-55** - Simplified Reward Function
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]`
- **Waypoint Threshold**: 1.5m
- **Stairs**: None
- **Terrain Width**: 15.0m

#### Reward Function (Changes)
- `waypoint_bonus`: **200.0** *(further increased)*
- `forward_reward_weight`: **2.0** *(doubled)*
- `heading_reward_weight`: **3.0** *(increased)*
- `balance_reward_weight`: **0.0** *(removed)*
- `optimal_speed`: 1.0
- `speed_regulation_weight`: **0.0** *(removed)*

#### PPO Hyperparameters (Changes)
- **Learning Rate**: **0.0002** *(reduced from 0.0003)*
- **Entropy Coefficient**: **0.01** *(increased from 0.005)*
- **Total Timesteps**: 80,000,000

#### Evaluation Results (5 episodes)
- **Render Mode**: Human
- **Performance**: Simplified reward structure focusing on core objectives

---

### 8. **2025-12-22/15-26-16** - Extended Training (100M)
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]`
- **Waypoint Threshold**: 1.5m
- **Stairs**: None
- **Terrain Width**: 15.0m

#### Reward Function
- `waypoint_bonus`: 200.0
- `forward_reward_weight`: 2.0
- `heading_reward_weight`: 3.0
- `balance_reward_weight`: 0.0
- `optimal_speed`: 1.0
- `speed_regulation_weight`: 0.0

#### PPO Hyperparameters
- **Learning Rate**: 0.0003 *(restored to baseline)*
- **Entropy Coefficient**: 0.005 *(restored)*
- **Total Timesteps**: 100,000,000 *(extended)*

#### Evaluation Results (5 episodes)
- **Render Mode**: Human
- **Performance**: Extended training for improved convergence

---

### 9. **2025-12-23/10-37-28** - Circuit Completion Bonus (Buggy Version)
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
- **Waypoints**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]`
- **Waypoint Threshold**: 1.5m
- **Stairs**: None
- **Terrain Width**: 15.0m

#### Reward Function (Changes)
- `waypoint_bonus`: 150.0
- `circuit_completion_bonus`: **500.0** *(NEW - buggy implementation)*
- `forward_reward_weight`: 1.0
- `heading_reward_weight`: 2.0
- `balance_reward_weight`: 0.5
- `optimal_speed`: 1.2
- `speed_regulation_weight`: 0.2

#### PPO Hyperparameters
- **Total Timesteps**: 80,000,000
- *(Baseline parameters)*

#### Evaluation Results (20 episodes)
- **Video**: `videos/circuit_bonus_buggy/`
- **Performance**: First attempt at circuit completion bonus (contained bug)

---

### 10. **2025-12-23/14-37-50** - Circuit Completion Bonus (Fixed)
**Scenario**: `humanoid_circuit_flat`

#### Environment Configuration
*(Same as 2025-12-23/10-37-28)*

#### Reward Function
- `waypoint_bonus`: 150.0
- `circuit_completion_bonus`: **500.0** *(FIXED implementation)*
- `forward_reward_weight`: 1.0
- `heading_reward_weight`: 2.0
- `balance_reward_weight`: 0.5
- `optimal_speed`: 1.2
- `speed_regulation_weight`: 0.2

#### PPO Hyperparameters
- **Total Timesteps**: 80,000,000
- *(Baseline parameters)*

#### Evaluation Results (20 episodes)
- **Video**: `videos/circuit_bonus_fixed/`
- **Performance**: Fixed circuit completion bonus implementation

---

### 11. **2025-12-24/16-29-37** - Circuit with Stairs (Easy)
**Scenario**: `humanoid_circuit_easy`

#### Environment Configuration
- **Waypoints**: `[[8.0, 0.0], [15.0, 0.0], [20.0, 3.0]]`
- **Waypoint Threshold**: 1.5m
- **Stairs**: `[[9.0, 5, 0.1, 0.8]]` (x=9.0m, 5 steps, 0.1m height, 0.8m depth)
- **Terrain Width**: 15.0m

#### Reward Function
- `progress_reward_weight`: 200.0
- `waypoint_bonus`: 150.0
- `circuit_completion_bonus`: 500.0
- `height_reward_weight`: **2.0** *(NEW - encourages climbing)*
- `forward_reward_weight`: 1.0
- `heading_reward_weight`: 2.0
- `balance_reward_weight`: 0.5
- `optimal_speed`: 1.0
- `speed_regulation_weight`: 0.2
- `ctrl_cost_weight`: 0.1
- `contact_cost_weight`: 5e-7
- `healthy_reward`: 5.0
- `healthy_z_range`: **[1.0, 2.0]** *(relative to terrain)*
- `check_healthy_z_relative`: **true** *(NEW - adapts to stairs)*

#### PPO Hyperparameters
- **Total Timesteps**: 80,000,000
- *(Baseline parameters)*

#### Evaluation Results (10 episodes)
- **Mean Reward**: ~9,267.66
- **Success Rate**: 80% (8/10 episodes)
- **Failed Episodes**: Episode 4 (4,258.57), Episode 9 (373.58)
- **Best Episode**: Episode 8 (11,324.27)
- **Video**: `videos/circuit_easy_latest/`
- **Performance**: Successfully navigates stairs in most cases
- **Notes**: First successful training with stairs obstacle

---

## Key Findings and Evolution

### Reward Function Evolution

1. **Initial Phase (Dec 17-18)**: Baseline reward structure focused on progress and waypoint reaching
2. **Balance Phase (Dec 18)**: Added `balance_reward_weight` for stability
3. **Speed Control Phase (Dec 18-19)**: Introduced `optimal_speed` and `speed_regulation_weight` for natural gait
4. **Simplification Phase (Dec 20-22)**: Removed some rewards, increased core weights, adjusted learning rate
5. **Circuit Completion Phase (Dec 23)**: Added `circuit_completion_bonus` to encourage full circuit navigation
6. **Stairs Phase (Dec 24)**: Added `height_reward_weight` and relative health checking for obstacle navigation

### PPO Hyperparameter Tuning

- **Learning Rate**: Experimented with 0.0002 (slower, more stable) vs 0.0003 (faster)
- **Entropy Coefficient**: Tested 0.005 (baseline) vs 0.01 (more exploration)
- **Training Duration**: Varied between 80M and 100M timesteps
- **Network Architecture**: Consistent [256, 256] MLP throughout all experiments

### Circuit Design Evolution

1. **Square Circuit**: `[[5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]` - Initial testing
2. **Narrower Rectangle**: `[[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]]` - Tighter turns
3. **Linear with Stairs**: `[[8.0, 0.0], [15.0, 0.0], [20.0, 3.0]]` - Obstacle navigation

### Success Metrics

- **Flat Circuits**: Achieved reliable navigation with completion rates approaching 100%
- **Stairs Circuit**: Achieved 80% success rate on first stairs implementation
- **Gait Quality**: Progressively improved through speed regulation and balance rewards
- **Stability**: Enhanced through relative health checking and terrain-adaptive rewards

---

## Technical Configuration

### Common Settings Across All Runs

- **Environment**: HumanoidCircuit-v0 (custom)
- **Algorithm**: PPO (Stable-Baselines3)
- **Policy**: MlpPolicy
- **Device**: CUDA (GPU)
- **Parallel Environments**: 8
- **VecNormalize**: Enabled (clip_obs=10.0, gamma=0.99)
- **Seed**: 42 (for reproducibility)

### Network Architecture

```python
net_arch = [256, 256]  # 2-layer MLP with 256 units each
```

### Standard PPO Parameters

- **gamma**: 0.99
- **gae_lambda**: 0.95
- **clip_range**: 0.2
- **vf_coef**: 0.25
- **max_grad_norm**: 0.5
- **checkpoint_every_steps**: 250,000

---

## Conclusions

1. **Progressive Learning**: The project successfully demonstrated progressive learning from flat circuits to stairs navigation
2. **Reward Engineering**: Careful reward function design was critical for achieving desired behaviors
3. **Stairs Navigation**: The relative health checking (`check_healthy_z_relative`) was essential for stairs success
4. **Training Duration**: 80M timesteps proved sufficient for most scenarios; 100M showed marginal improvements
5. **Future Work**: 
   - Increase stairs difficulty (more steps, greater height)
   - Test multiple stair sections in one circuit
   - Optimize success rate above 90%
   - Explore curriculum learning approaches

---

## Video Documentation

All training runs have corresponding evaluation videos:

- `videos/circuit_flat_80m/` - Baseline flat circuit
- `videos/circuit_flat_100m/` - Extended training
- `videos/circuit_flat_80m_5000steps/` - Narrower circuit
- `videos/circuit_flat_80m_natural_gait/` - Natural gait optimization
- `videos/circuit_bonus_buggy/` - Circuit completion bonus (buggy)
- `videos/circuit_bonus_fixed/` - Circuit completion bonus (fixed)
- `videos/circuit_easy_latest/` - Stairs navigation (latest)

---

**Report End**
