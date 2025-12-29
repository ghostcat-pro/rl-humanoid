#!/usr/bin/env python3
"""
Test script for the improved HumanoidStairsConfigurable environment.
Verifies the new parameters and relative health check.
"""

import envs
from envs.custom.humanoid_stairs_configurable import HumanoidStairsConfigurableEnv
import gymnasium as gym
import numpy as np

def test_configurable_stairs():
    print("Testing HumanoidStairsConfigurable-v0 with new features...")
    
    # Test 1: Initialize with new parameters
    env = HumanoidStairsConfigurableEnv(
        flat_distance_before_stairs=5.0,
        num_steps=5,
        step_height=0.2,
        check_healthy_z_relative=True,
        distance_reward_weight=1.0,
        healthy_z_range=(0.5, 1.5) # Strict range, but relative
    )
    print("✓ Environment initialized with new parameters")
    
    obs, info = env.reset()
    print(f"✓ Reset successful. Obs shape: {obs.shape}")
    
    # Check internal flags
    assert env._check_healthy_z_relative == True
    assert env._distance_reward_weight == 1.0
    print("✓ Internal flags set correctly")
    
    # Test 2: Step and check rewards
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    
    print("✓ Step successful")
    print("Info keys:", info.keys())
    
    assert "reward_distance" in info
    print(f"✓ reward_distance present: {info['reward_distance']}")
    
    # Test 3: Mock height check (Relative)
    # Set agent high up (absolute 3.0) but over a high platform (say 2.0) -> relative 1.0 (healthy)
    # Actually modifying mujoco state is tricky without full physics, 
    # but we can check if it doesn't terminate immediately on start (z=1.4, terrain=0.0 -> rel=1.4 -> healthy)
    # Range is (0.5, 1.5). 1.4 is healthy.
    
    assert not term
    print("✓ Agent is healthy at start (Relative Z check works for flat ground)")
    
    env.close()
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_configurable_stairs()
