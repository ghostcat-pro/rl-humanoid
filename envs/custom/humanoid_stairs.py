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
        
        # Track max step reached
        self._max_step_reached = 0
        self._steps_x_bounds = []
        # Stairs start at x=20.0 (start of first step) and each is 0.6m long
        # Stair 1 center: 20.3 -> range [20.0, 20.6]
        # Stair 2 center: 20.9 -> range [20.6, 21.2]
        # ...
        for i in range(10):
            start_x = 20.0 + i * 0.6
            end_x = start_x + 0.6
            self._steps_x_bounds.append((start_x, end_x))

        # Path to our custom XML file
        xml_file = os.path.join(
            os.path.dirname(__file__), "..", "assets", "humanoid_stairs.xml"
        )

        # Terrain perception: 5x5 grid around agent = 25 height values
        # Agent is at center of grid for symmetric perception
        # Original observation: 376 dimensions
        # + 25 for height grid
        # = 401 total dimensions
        self._height_grid_size = 5  # 5x5 grid (agent at center)
        self._height_sample_distance = 0.3  # 0.3m spacing between samples
        height_grid_dims = self._height_grid_size * self._height_grid_size
        
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376 + height_grid_dims,), dtype=np.float64
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

    def _get_terrain_height_at(self, x, y):
        """
        Get terrain height at a specific (x, y) position.
        
        Returns the expected height based on stair geometry:
        - Platform (x < 20.0): height = 0
        - Stairs (20.0 <= x < 26.0): height increases by 0.15m per step
        - End platform (x >= 26.0): height = 1.5m
        """
        if x < 20.0:
            # Starting platform (20m of flat terrain)
            return 0.0
        elif x >= 26.0:
            # End platform at top of stairs
            return 1.5
        else:
            # On stairs - calculate which step
            step_index = int((x - 20.0) / 0.6)  # Each step is 0.6m long
            step_index = min(step_index, 9)  # Max 10 steps (0-9)
            return step_index * 0.15
    
    def _get_height_grid(self):
        """
        Sample terrain heights in a grid around the agent.
        
        Creates a 5x5 grid centered on the agent's current position,
        with samples spaced 0.3m apart. This gives the agent local
        terrain awareness with the agent at the exact center.
        
        Grid layout (top view):
            • • • • •     Grid covers 1.2m x 1.2m
            • • • • •     Agent (A) is at center
            • • A • •     25 total sample points
            • • • • •     0.3m spacing
            • • • • •
        
        Returns:
            np.ndarray: Flattened array of 25 height values (relative to agent's current height)
        """
        current_x = self.data.qpos[0]
        current_y = self.data.qpos[1]
        current_z = self.data.qpos[2]
        
        heights = []
        
        # Create grid: 5x5 centered on agent
        # Grid spans from -0.6m to +0.6m in both x and y (1.2m total)
        grid_half_size = (self._height_grid_size - 1) * self._height_sample_distance / 2
        
        for i in range(self._height_grid_size):
            for j in range(self._height_grid_size):
                # Calculate offset from center
                dx = -grid_half_size + i * self._height_sample_distance
                dy = -grid_half_size + j * self._height_sample_distance
                
                # Sample position
                sample_x = current_x + dx
                sample_y = current_y + dy
                
                # Get terrain height at this position
                terrain_height = self._get_terrain_height_at(sample_x, sample_y)
                
                # Store relative height (terrain height - current agent height)
                relative_height = terrain_height - current_z
                heights.append(relative_height)
        
        return np.array(heights, dtype=np.float64)

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        # Get local height grid for terrain perception
        height_grid = self._get_height_grid()

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
                height_grid,  # Add terrain perception
            )
        )

    def _update_stair_colors(self, current_x):
        """Update stair colors based on progress."""
        # Default color: rgba="0.7 0.8 0.7 1" (light green-ish)
        # Active color: rgba="0.2 0.8 0.2 1" (bright green)
        
        for i, (start_x, end_x) in enumerate(self._steps_x_bounds):
            geom_name = f"stair_{i+1}"
            try:
                geom_id = self.model.geom(geom_name).id
                if start_x <= current_x:
                    # Passed or on this step - make it bright
                    self.model.geom_rgba[geom_id] = [0.2, 0.9, 0.2, 1.0]
                else:
                    # Not reached - default
                    self.model.geom_rgba[geom_id] = [0.7, 0.8, 0.7, 1.0]
            except KeyError:
                pass

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
        
        # Upright reward (alignment of torso z-axis with global z-axis)
        # qpos[3:7] is the quaternion (w, x, y, z) of the root (torso)
        # We want z-axis of torso to be close to global z.
        # Actually, let's use the z-component of the torso's z-axis vector.
        # But simpler: just penalize large deviations in pitch/roll.
        # Or use the projection of the torso's up vector onto the global z axis.
        # self.data.ximat[1] gives rotation matrix of body 1 (torso).
        # ximat is 3x3 flattened? No, ximat is nbody x 9.
        # Let's stick to a simpler heuristic or use the one from reward_wrappers if available,
        # but here we are inside the env.
        # Let's use a simple penalty for non-uprightness if we can easily compute it.
        # Alternatively, just increase healthy reward if z is high.
        
        # Let's add a "Step Reward" for reaching a new step
        step_reward = 0.0
        current_step = 0
        for i, (start_x, end_x) in enumerate(self._steps_x_bounds):
            if xy_position[0] >= start_x:
                current_step = i + 1
        
        if current_step > self._max_step_reached:
            step_reward = 10.0 * (current_step - self._max_step_reached)
            self._max_step_reached = current_step
            
        # Stuck penalty
        stuck_penalty = 0.0
        if not self.terminated and np.linalg.norm(xy_velocity) < 0.1:
            stuck_penalty = -0.1

        # Control cost
        ctrl_cost = self.control_cost(action)

        # Total reward
        reward = forward_reward + height_reward + healthy_reward + step_reward + stuck_penalty - ctrl_cost

        # Observation and termination
        observation = self._get_obs()
        terminated = self.terminated
        
        # Update visualization
        self._update_stair_colors(xy_position[0])

        info = {
            "reward_forward": forward_reward,
            "reward_height": height_reward,
            "reward_survive": healthy_reward,
            "reward_step": step_reward,
            "reward_stuck": stuck_penalty,
            "cost_ctrl": ctrl_cost,
            "x_position": xy_position[0],
            "y_position": xy_position[1],
            "z_position": z_position,
            "x_velocity": xy_velocity[0],
            "distance_from_origin": np.linalg.norm(xy_position, ord=2),
            "current_step": current_step,
            "max_step_reached": self._max_step_reached,
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
        
        self._max_step_reached = 0
        self._update_stair_colors(0.0)

        observation = self._get_obs()
        return observation
