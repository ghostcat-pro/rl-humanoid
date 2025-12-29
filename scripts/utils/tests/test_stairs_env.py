#!/usr/bin/env python3
"""
Quick test script to verify HumanoidStairs environment is working correctly.
"""

import envs  # Register custom environments
import gymnasium as gym
import numpy as np

def test_stairs_environment():
    """Test that the HumanoidStairs environment can be created and used."""
    print("Testing HumanoidStairs-v0 environment...")
    print("-" * 60)

    # Create environment
    env = gym.make("HumanoidStairs-v0")
    print(f"✓ Environment created successfully")

    # Check spaces
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    obs, info = env.reset()
    print(f"\n✓ Environment reset successfully")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Expected shape: (376,)")
    assert obs.shape == (376,), f"Unexpected observation shape: {obs.shape}"

    # Take a few random steps
    print(f"\n✓ Taking 10 random steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {i+1}")
            break

    print(f"✓ Steps completed successfully")
    print(f"Total reward from 10 steps: {total_reward:.2f}")
    print(f"Final x position: {info.get('x_position', 'N/A'):.2f}")
    print(f"Final z position (height): {info.get('z_position', 'N/A'):.2f}")

    env.close()
    print("\n" + "=" * 60)
    print("All tests passed! HumanoidStairs environment is working.")
    print("=" * 60)

if __name__ == "__main__":
    test_stairs_environment()
