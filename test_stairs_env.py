"""Test script for HumanoidStairs environment."""

import gymnasium as gym
import numpy as np

# Register the custom environment
import envs

def test_environment():
    """Test basic functionality of the HumanoidStairs environment."""
    print("Creating HumanoidStairs-v0 environment...")
    env = gym.make("HumanoidStairs-v0")

    # Get the unwrapped environment to access MuJoCo data
    unwrapped_env = env.unwrapped

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial position (x, y, z): ({unwrapped_env.data.qpos[0]:.3f}, {unwrapped_env.data.qpos[1]:.3f}, {unwrapped_env.data.qpos[2]:.3f})")

    print("\nRunning 10 random steps...")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {step+1}: reward={reward:.2f}, x={info['x_position']:.2f}, z={info['z_position']:.2f}, terminated={terminated}")

        if terminated:
            print("Episode terminated!")
            break

    print(f"\nTotal reward: {total_reward:.2f}")

    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"After reset - position (x, y, z): ({unwrapped_env.data.qpos[0]:.3f}, {unwrapped_env.data.qpos[1]:.3f}, {unwrapped_env.data.qpos[2]:.3f})")

    env.close()
    print("\n✅ Environment test completed successfully!")

if __name__ == "__main__":
    test_environment()
