"""Test script to verify configurable stairs environment works properly."""

import gymnasium as gym
import envs  # Register custom environments
import numpy as np


def test_config(config_name, config_params):
    """Test a specific stairs configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    # Create environment
    env = gym.make("HumanoidStairsConfigurable-v0", **config_params)
    
    # Print configuration
    print(f"Flat distance: {env._flat_distance}m")
    print(f"Steps up: {env._num_steps}")
    print(f"Step height: {env._step_height}m")
    print(f"Step depth: {env._step_depth}m")
    print(f"End platform: {env._end_platform_length}m")
    print(f"Stairs down: {env._num_steps_down if env._stairs_after_top else 0}")
    print(f"Abyss ending: {env._end_with_abyss}")
    print(f"\nTotal height: {env._max_height}m")
    print(f"Stairs range: {env._stairs_start_x}m to {env._stairs_end_x}m")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Test reset and step
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial position: x={env.data.qpos[0]:.2f}, z={env.data.qpos[2]:.2f}")
    
    # Take a few random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i == 0:
            print(f"\nAfter first step:")
            print(f"  Position: x={info['x_position']:.2f}, z={info['z_position']:.2f}")
            print(f"  Reward: {reward:.2f}")
    
    env.close()
    print(f"✓ {config_name} test passed!")


if __name__ == "__main__":
    print("Testing Configurable Stairs Environment")
    print("=" * 60)
    
    # Test 1: Standard configuration
    test_config("Standard Stairs", {
        "flat_distance_before_stairs": 20.0,
        "num_steps": 10,
        "step_height": 0.15,
        "step_depth": 0.6,
        "end_platform_length": 5.0,
        "stairs_after_top": False,
        "end_with_abyss": False,
    })
    
    # Test 2: Easy stairs
    test_config("Easy Stairs", {
        "flat_distance_before_stairs": 20.0,
        "num_steps": 8,
        "step_height": 0.10,
        "step_depth": 0.8,
        "end_platform_length": 5.0,
        "stairs_after_top": False,
        "end_with_abyss": False,
    })
    
    # Test 3: Hard stairs
    test_config("Hard Stairs", {
        "flat_distance_before_stairs": 15.0,
        "num_steps": 15,
        "step_height": 0.20,
        "step_depth": 0.4,
        "end_platform_length": 3.0,
        "stairs_after_top": False,
        "end_with_abyss": False,
    })
    
    # Test 4: Abyss ending
    test_config("Abyss Ending", {
        "flat_distance_before_stairs": 20.0,
        "num_steps": 10,
        "step_height": 0.15,
        "step_depth": 0.6,
        "end_platform_length": 0.0,
        "stairs_after_top": False,
        "end_with_abyss": True,
    })
    
    # Test 5: Up and down
    test_config("Up and Down", {
        "flat_distance_before_stairs": 20.0,
        "num_steps": 10,
        "step_height": 0.15,
        "step_depth": 0.6,
        "end_platform_length": 3.0,
        "stairs_after_top": True,
        "num_steps_down": 8,
        "end_with_abyss": False,
    })
    
    # Test 6: Tiny steps
    test_config("Tiny Steps", {
        "flat_distance_before_stairs": 10.0,
        "num_steps": 20,
        "step_height": 0.075,
        "step_depth": 0.3,
        "end_platform_length": 5.0,
        "stairs_after_top": False,
        "end_with_abyss": False,
    })
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
