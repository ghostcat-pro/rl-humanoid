"""
Example: Training Humanoid-v5 with DestinationWrapper

This demonstrates using the wrapper approach instead of a custom environment.
This does NOT interfere with the existing HumanoidDestination-v0 environment.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.destination_wrapper import DestinationWrapper


# Example 1: Wrap standard Humanoid-v5
def make_wrapped_humanoid():
    """Create standard Humanoid with destination wrapper."""
    env = gym.make("Humanoid-v5")
    env = DestinationWrapper(
        env,
        target_position=(5.0, 0.0),
        distance_reward_weight=1.0,
        reach_reward_weight=10.0,
        reach_threshold=0.5,
    )
    return env


# Example 2: Vectorized environment
def make_vec_env(n_envs=8):
    """Create vectorized wrapped environments."""
    def make_env():
        return make_wrapped_humanoid()
    
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    return vec_env


if __name__ == "__main__":
    # Create environment
    env = make_vec_env(n_envs=8)
    
    # Train with PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log="./outputs_wrapper/",
    )
    
    # Train
    model.learn(total_timesteps=1_000_000)
    
    # Save
    model.save("humanoid_wrapper_destination")
    env.save("humanoid_wrapper_vecnorm.pkl")
    
    print("\n✓ Wrapper-based training complete!")
    print("This is separate from HumanoidDestination-v0 environment")
