from gymnasium.envs.registration import register

register(
    id="HumanoidStairs-v0",
    entry_point="envs.humanoid_stairs:HumanoidStairsEnv",
    max_episode_steps=1000,
)
