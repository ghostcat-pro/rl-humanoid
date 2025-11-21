from gymnasium.envs.registration import register

register(
    id="HumanoidStairs-v0",
    entry_point="envs.humanoid_stairs:HumanoidStairsEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidDestination-v0",
    entry_point="envs.humanoid_destination:HumanoidDestinationEnv",
    max_episode_steps=1000,
)
