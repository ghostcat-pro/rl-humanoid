#!/usr/bin/env python3
"""Quick test to verify destination termination works correctly."""

import gymnasium as gym
import numpy as np

# Import environment registration
import envs

def test_termination_on():
    """Test with termination enabled."""
    print("\n=== Test 1: Termination ON ===")
    env = gym.make(
        "HumanoidDestination-v0",
        target_position=(2.0, 0.0),  # Close target for quick testing
        terminate_at_destination=True,
        destination_threshold=0.5,
    )
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(1000):
        # Simple forward action to move toward target
        action = np.zeros(17)
        action[0] = 1.0  # Move forward
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps % 50 == 0:
            print(f"  Step {steps}: Distance={info['distance_to_target']:.2f}m, "
                  f"Reached={info['reached_destination']}")
        
        if terminated or truncated:
            print(f"  ✓ Episode ended at step {steps}")
            print(f"  Final distance: {info['distance_to_target']:.2f}m")
            print(f"  Reached destination: {info['reached_destination']}")
            print(f"  Total reward: {total_reward:.2f}")
            break
    else:
        print(f"  ✗ Episode ran full 1000 steps without terminating")
    
    env.close()

def test_termination_off():
    """Test with termination disabled."""
    print("\n=== Test 2: Termination OFF ===")
    env = gym.make(
        "HumanoidDestination-v0",
        target_position=(2.0, 0.0),
        terminate_at_destination=False,
        destination_threshold=0.5,
    )
    
    obs, info = env.reset()
    reached_count = 0
    steps = 0
    
    for _ in range(1000):
        action = np.zeros(17)
        action[0] = 1.0
        
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        if info['reached_destination']:
            reached_count += 1
        
        if steps % 50 == 0:
            print(f"  Step {steps}: Distance={info['distance_to_target']:.2f}m, "
                  f"Times reached={reached_count}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {steps} (health issue)")
            break
    else:
        print(f"  ✓ Episode ran full 1000 steps")
        print(f"  Times reached destination: {reached_count}")
    
    env.close()

def test_with_config():
    """Test using the default config values."""
    print("\n=== Test 3: Using Default Config ===")
    env = gym.make("HumanoidDestination-v0")
    
    obs, info = env.reset()
    print(f"  Target position: {env.unwrapped._target_position}")
    print(f"  Terminate at destination: {env.unwrapped._terminate_at_destination}")
    print(f"  Destination threshold: {env.unwrapped._destination_threshold}m")
    
    env.close()

if __name__ == "__main__":
    print("Testing HumanoidDestination-v0 termination behavior")
    print("=" * 60)
    
    test_with_config()
    test_termination_on()
    test_termination_off()
    
    print("\n" + "=" * 60)
    print("✓ All tests completed!")
