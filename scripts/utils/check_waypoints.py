#!/usr/bin/env python3
"""Check detailed waypoint reaching information."""

import sys
sys.path.insert(0, '/home/fspinto/projects/rl-humanoid')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import envs  # Register custom environments

# Load model
model_path = "outputs/2025-12-23/14-37-50/eval/best_model.zip"
vecnorm_path = "outputs/2025-12-23/14-37-50/vecnormalize_final.pkl"

env_kwargs = {
    "waypoints": [[5.0, 0.0], [5.0, 5.0], [2.0, 5.0], [2.0, 0.0]],
    "waypoint_reach_threshold": 1.5,
    "stairs": [],
    "terrain_width": 15.0,
    "progress_reward_weight": 200.0,
    "waypoint_bonus": 150.0,
    "circuit_completion_bonus": 500.0,
    "height_reward_weight": 0.0,
    "forward_reward_weight": 1.0,
    "heading_reward_weight": 2.0,
    "balance_reward_weight": 0.5,
    "optimal_speed": 1.2,
    "speed_regulation_weight": 0.2,
    "ctrl_cost_weight": 0.1,
    "contact_cost_weight": 5e-7,
    "healthy_reward": 5.0,
    "terminate_when_unhealthy": True,
    "healthy_z_range": [0.8, 3.0],
}

# Create environment
def make_env():
    return gym.make("HumanoidCircuit-v0", **env_kwargs)

env = DummyVecEnv([make_env])
env = VecNormalize.load(vecnorm_path, env)
env.training = False
env.norm_reward = False

# Load model
model = PPO.load(model_path, env=env)

# Run episodes
num_episodes = 5
for ep in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    waypoints_reached = 0
    circuit_complete = False
    waypoint_rewards = 0
    circuit_bonus_collected = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        steps += 1
        
        # Track waypoint reaching
        if info[0].get('waypoint_just_reached', False):
            waypoints_reached += 1
        
        if info[0].get('circuit_complete', False) and not circuit_complete:
            circuit_complete = True
            
        # Track specific rewards
        waypoint_rewards += info[0].get('reward_waypoint', 0)
        circuit_bonus_collected += info[0].get('reward_circuit_completion', 0)
        
        if done[0]:
            break
    
    print(f"\n{'='*60}")
    print(f"Episode {ep+1}:")
    print(f"{'='*60}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Steps: {steps}")
    print(f"Waypoints Reached: {waypoints_reached}/4")
    print(f"Circuit Completed: {'YES' if circuit_complete else 'NO'}")
    print(f"Waypoint Bonus Collected: {waypoint_rewards:.2f}")
    print(f"Circuit Completion Bonus: {circuit_bonus_collected:.2f}")
    print(f"Other Rewards: {total_reward - waypoint_rewards - circuit_bonus_collected:.2f}")
    
    # Reward breakdown
    print(f"\nReward Sources:")
    print(f"  Waypoint bonuses: {waypoint_rewards:.2f}")
    print(f"  Circuit bonus: {circuit_bonus_collected:.2f}")
    print(f"  Per-step rewards: {total_reward - waypoint_rewards - circuit_bonus_collected:.2f}")
    print(f"  Average per step: {(total_reward - waypoint_rewards - circuit_bonus_collected) / steps:.2f}")

env.close()
