"""Test circuit environment."""

import gymnasium as gym
import envs
import numpy as np


def test_circuit():
    """Test circuit environment with simple configuration."""
    print("="*60)
    print("Testing Circuit Environment")
    print("="*60)
    
    # Create environment
    env = gym.make(
        "HumanoidCircuit-v0",
        waypoints=[(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
        stairs=[(8.0, 5, 0.15, 0.6)],
        waypoint_reach_threshold=1.0,
    )
    
    print(f"\nEnvironment: HumanoidCircuit-v0")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Waypoints: {len(env._waypoints)}")
    print(f"Stairs sections: {len(env._stairs)}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial position: x={env.data.qpos[0]:.2f}, y={env.data.qpos[1]:.2f}, z={env.data.qpos[2]:.2f}")
    print(f"Current waypoint: {env.current_waypoint}")
    print(f"Current waypoint index: {env._current_waypoint_index}")
    
    # Take some steps
    print("\nTaking 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i == 0 or i == 9:
            print(f"\nStep {i+1}:")
            print(f"  Position: ({info['x_position']:.2f}, {info['y_position']:.2f}, {info['z_position']:.2f})")
            print(f"  Distance to waypoint: {info['distance_to_waypoint']:.2f}m")
            print(f"  Current waypoint index: {info['current_waypoint_index']}")
            print(f"  Waypoints reached: {info['waypoints_reached']}")
            print(f"  Reward: {reward:.2f}")
            print(f"    - Progress: {info['reward_progress']:.2f}")
            print(f"    - Forward: {info['reward_forward']:.2f}")
            print(f"    - Height: {info['reward_height']:.2f}")
            print(f"    - Waypoint bonus: {info['reward_waypoint']:.2f}")
        
        if terminated:
            print(f"\n  Terminated at step {i+1}")
            break
    
    env.close()
    print("\n" + "="*60)
    print("✓ Circuit environment test passed!")
    print("="*60)


if __name__ == "__main__":
    test_circuit()
