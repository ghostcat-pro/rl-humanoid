"""Visual test for HumanoidStairs environment - displays the stairs."""

import gymnasium as gym
import envs

def visual_test():
    """Test environment with rendering to visualize stairs."""
    print("Creating HumanoidStairs-v0 with rendering...")
    env = gym.make("HumanoidStairs-v0", render_mode="human")

    print("Resetting environment...")
    obs, info = env.reset()

    print("Running 500 steps with random actions...")
    print("You should see the humanoid and the stairs!")
    print("Press Ctrl+C to stop early.")

    try:
        for step in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 50 == 0:
                print(f"Step {step}: x={info['x_position']:.2f}, z={info['z_position']:.2f}")

            if terminated or truncated:
                print(f"Episode ended at step {step}")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nStopped by user")

    env.close()
    print("Visual test complete!")

if __name__ == "__main__":
    visual_test()
