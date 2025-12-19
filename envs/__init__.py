"""Custom environment registration."""

from gymnasium.envs.registration import register

# Import custom environments to make them available
from envs.custom import (
    HumanoidStairsEnv,
    HumanoidDestinationEnv,
    HumanoidStairsConfigurableEnv,
    HumanoidCircuitEnv,
)

register(
    id="HumanoidStairs-v0",
    entry_point="envs.custom.humanoid_stairs:HumanoidStairsEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidDestination-v0",
    entry_point="envs.custom.humanoid_destination:HumanoidDestinationEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStairsConfigurable-v0",
    entry_point="envs.custom.humanoid_stairs_configurable:HumanoidStairsConfigurableEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidCircuit-v0",
    entry_point="envs.custom.humanoid_circuit:HumanoidCircuitEnv",
    max_episode_steps=5000,  # Increased for slower, controlled navigation with balance
)

__all__ = [
    "HumanoidStairsEnv",
    "HumanoidDestinationEnv",
    "HumanoidStairsConfigurableEnv",
    "HumanoidCircuitEnv",
]
