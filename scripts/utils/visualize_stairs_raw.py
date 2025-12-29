"""Visualize stairs environment without model - just to see the setup."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom environments to register them
import envs
import gymnasium as gym

# Create environment with exact training configuration
env = gym.make(
    "HumanoidStairsConfigurable-v0",
    render_mode="human",
    flat_distance_before_stairs=1.0,
    num_steps=8,
    step_height=0.1,
    step_depth=0.8,
    end_platform_length=5.0,
    stairs_after_top=False,
    end_with_abyss=False,
)

print("=" * 70)
print("STAIRS VISUALIZATION - No Model")
print("Configuration:")
print("  - Flat distance before stairs: 1.0m")
print("  - Number of steps: 8")
print("  - Step height: 0.1m (10cm)")
print("  - Step depth: 0.8m (80cm)")
print()
print("The humanoid will take RANDOM actions.")
print("Look for the stairs - they should be 1 meter in front!")
print("Use mouse to rotate/zoom camera in the viewer window.")
print("=" * 70)

# Run with random actions to show the environment
for episode in range(3):
    obs, info = env.reset()
    print(f"\nEpisode {episode + 1} starting...")
    
    for step in range(200):  # Short episodes just to show the environment
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break
    else:
        print(f"  Episode completed 200 steps")

env.close()
print("\nVisualization complete!")
