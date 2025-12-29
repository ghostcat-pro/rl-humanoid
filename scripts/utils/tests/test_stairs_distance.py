"""Quick test to visualize stairs at diffe)

# Load the trained model
model = PPO.load(MODEL_PATH, device="cuda")
env = VecNormalize.load(VECNORM_PATH, env)tances."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom environments to register them
import envs

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from utils.make_env import make_vector_env

# Configuration
NUM_EPISODES = 5  # Can be changed to any number
MODEL_PATH = "outputs/2025-11-23/14-26-39/eval/best_model.zip"
VECNORM_PATH = "outputs/2025-11-23/14-26-39/vecnormalize_final.pkl"

# Create environment with 1m flat distance - EXACTLY matching training config
env = make_vector_env(
    env_id="HumanoidStairsConfigurable-v0",
    n_envs=1,
    start_method="spawn",
    monitor=True,
    make_kwargs={
        "render_mode": "human",
        "flat_distance_before_stairs": 1.0,  # 1 meter!
        "num_steps": 8,
        "step_height": 0.1,
        "step_depth": 0.8,
        "end_platform_length": 5.0,
        "stairs_after_top": False,
        "end_with_abyss": False,
        "forward_reward_weight": 2.0,
        "height_reward_weight": 5.0,
        "ctrl_cost_weight": 0.05,
        "contact_cost_weight": 2e-7,
        "healthy_reward": 5.0,
        "step_bonus": 25.0,
        "terminate_when_unhealthy": True,
        "healthy_z_range": (0.8, 3.0),
    }
)

# Load the trained model
model_path = "outputs/2025-11-23/13-28-13/final_model.zip"
vecnorm_path = "outputs/2025-11-23/13-28-13/vecnormalize_final.pkl"

model = PPO.load(model_path, device="cuda")
env = VecNormalize.load(vecnorm_path, env)
env.training = False
env.norm_reward = False

print("=" * 60)
print("Stairs positioned at 1 METER from starting position")
print("The humanoid should encounter stairs almost immediately!")
print(f"Running {NUM_EPISODES} episodes...")
print("=" * 60)

# Run episodes
for ep in range(NUM_EPISODES):
    obs = env.reset()
    episode_reward = 0
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        steps += 1
        
        if done[0]:
            print(f"Episode {ep+1}: reward={episode_reward:.2f}, steps={steps}")
            break

env.close()
