"""Custom Humanoid environment with stairs for climbing task."""

from typing import Dict, Any
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os


class HumanoidStairsEnv(MujocoEnv, utils.EzPickle):
    """
    Humanoid environment with stairs.

    The agent must learn to climb stairs going in the positive x direction.
    Stairs start at x=1.5 and end at x=7.5, with 10 steps each 0.15m high.

    Reward components:
    - Forward progress (positive x velocity)
    - Height gained (positive z position)
    - Staying alive
    - Penalties for falling or excessive control
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        forward_reward_weight=1.25,
        height_reward_weight=2.0,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 3.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            height_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._height_reward_weight = height_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # Track initial and previous position for reward calculation
        self._init_position = None
        self._prev_position = None
        self._prev_height = None

        # Path to our custom XML file
        xml_file = os.path.join(
            os.path.dirname(__file__), "assets", "humanoid_stairs.xml"
        )

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,  # frame_skip
            observation_space=observation_space,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return float(self.is_healthy) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        # Store position before step
        prev_xy_position = self.data.qpos[0:2].copy()
        prev_z_position = self.data.qpos[2]

        # Take the action
        self.do_simulation(action, self.frame_skip)

        # Get new position
        xy_position = self.data.qpos[0:2].copy()
        z_position = self.data.qpos[2]
        xy_velocity = self.data.qvel[0:2].copy()

        # Calculate rewards
        # Forward progress reward (positive x direction)
        forward_reward = self._forward_reward_weight * xy_velocity[0]

        # Height reward - reward for gaining height
        height_gained = z_position - prev_z_position
        height_reward = self._height_reward_weight * max(0, height_gained)

        # Healthy reward for staying upright
        healthy_reward = self.healthy_reward

        # Control cost
        ctrl_cost = self.control_cost(action)

        # Total reward
        reward = forward_reward + height_reward + healthy_reward - ctrl_cost

        # Observation and termination
        observation = self._get_obs()
        terminated = self.terminated

        info = {
            "reward_forward": forward_reward,
            "reward_height": height_reward,
            "reward_survive": healthy_reward,
            "cost_ctrl": ctrl_cost,
            "x_position": xy_position[0],
            "y_position": xy_position[1],
            "z_position": z_position,
            "x_velocity": xy_velocity[0],
            "distance_from_origin": np.linalg.norm(xy_position, ord=2),
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        # Start the agent at the beginning of the stairs
        qpos[0] = 0.0  # x position - start on platform
        qpos[1] = 0.0  # y position - centered
        qpos[2] = 1.4  # z position - standing height

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
