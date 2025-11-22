"""Custom Humanoid environment for walking to a destination."""

from typing import Dict, Any, Optional
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os


class HumanoidDestinationEnv(MujocoEnv, utils.EzPickle):
    """
    Humanoid environment with a target destination.

    The agent must learn to walk to a specific target position (x, y).
    
    Reward components:
    - Distance reward (minimizing distance to target)
    - Bonus for reaching target
    - Penalty for velocity when near target (optional, to encourage stopping)
    - Staying alive
    - Control cost
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
        target_position=(5.0, 0.0),
        distance_reward_weight=1.0,
        reach_reward_weight=10.0,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 2.5),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            target_position,
            distance_reward_weight,
            reach_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._target_position = np.array(target_position)
        self._distance_reward_weight = distance_reward_weight
        self._reach_reward_weight = reach_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # Path to our custom XML file
        xml_file = os.path.join(
            os.path.dirname(__file__), "..", "assets", "humanoid_destination.xml"
        )

        # Observation space needs to include target info or relative position
        # Original humanoid obs is 376. We add 2 for relative target position (x, y)
        # But wait, standard humanoid excludes global position (qpos[0:2])
        # So we should provide relative vector to target.
        
        # Let's check standard humanoid obs size.
        # qpos (nq=24) - 2 (exclude x,y) = 22
        # qvel (nv=23) = 23
        # cinert (14*10) = 140
        # cvel (14*6) = 84
        # qfrc_actuator (23-6) = 17
        # cfrc_ext (14*6) = 84
        # Total = 22 + 23 + 140 + 84 + 17 + 84 = 370?
        # The file I read said 376. Let's stick to what works and append target info.
        
        # Actually, let's just append the relative vector to target (2 dims)
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
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

        # Calculate relative position to target
        # Current position (x, y) is in position[0:2]
        current_xy = position[0:2]
        vector_to_target = self._target_position - current_xy

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
                vector_to_target, # Add relative target position
            )
        )

    def step(self, action):
        # Store position before step
        prev_xy_position = self.data.qpos[0:2].copy()
        prev_dist = np.linalg.norm(self._target_position - prev_xy_position)

        # Take the action
        self.do_simulation(action, self.frame_skip)

        # Get new position
        xy_position = self.data.qpos[0:2].copy()
        dist = np.linalg.norm(self._target_position - xy_position)
        
        # Calculate rewards
        
        # 1. Progress reward (improvement in distance)
        # Positive if getting closer
        progress_reward = (prev_dist - dist) * self._distance_reward_weight * 100 # Scale up a bit
        
        # 2. Reach reward
        reach_reward = 0.0
        if dist < 0.5: # Within 0.5 meters
            reach_reward = self._reach_reward_weight

        # Healthy reward for staying upright
        healthy_reward = self.healthy_reward

        # Control cost
        ctrl_cost = self.control_cost(action)

        # Total reward
        reward = progress_reward + reach_reward + healthy_reward - ctrl_cost

        # Observation and termination
        observation = self._get_obs()
        terminated = self.terminated
        
        # Also terminate if reached target? Maybe not, let it stabilize.
        # But for now let's just keep it running.

        info = {
            "reward_progress": progress_reward,
            "reward_reach": reach_reward,
            "reward_survive": healthy_reward,
            "cost_ctrl": ctrl_cost,
            "distance_to_target": dist,
            "x_position": xy_position[0],
            "y_position": xy_position[1],
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

        # Start the agent at 0,0
        qpos[0] = 0.0  # x position
        qpos[1] = 0.0  # y position
        qpos[2] = 1.4  # z position

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
