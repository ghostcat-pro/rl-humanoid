"""Custom humanoid environments."""

from envs.custom.humanoid_stairs import HumanoidStairsEnv
from envs.custom.humanoid_destination import HumanoidDestinationEnv
from envs.custom.humanoid_stairs_configurable import HumanoidStairsConfigurableEnv
from envs.custom.humanoid_circuit import HumanoidCircuitEnv

__all__ = [
    "HumanoidStairsEnv",
    "HumanoidDestinationEnv", 
    "HumanoidStairsConfigurableEnv",
    "HumanoidCircuitEnv",
]
