"""Humanoid Circuit Environment - Navigate through waypoints with obstacles."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import tempfile


class HumanoidCircuitEnv(MujocoEnv, utils.EzPickle):
    """
    Humanoid environment with sequential waypoint navigation and configurable terrain.
    
    The agent must visit a sequence of waypoints in order, potentially navigating
    over stairs and other obstacles. Combines navigation and locomotion skills.
    
    Features:
    - Sequential waypoint navigation (must visit in order)
    - Configurable stairs between waypoints
    - Height grid for terrain perception
    - Rewards for waypoint progress and completion
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
        # Waypoint configuration
        waypoints: List[Tuple[float, float]] = [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)],
        waypoint_reach_threshold: float = 1.0,
        
        # Stairs configuration (list of stair sections)
        # Each stair: (start_x, num_steps, step_height, step_depth)
        stairs: List[Tuple[float, int, float, float]] = [(8.0, 5, 0.15, 0.6), (28.0, 5, 0.15, 0.6)],
        
        # Terrain configuration
        terrain_width: float = 4.0,  # Width of the terrain (y-axis)
        
        # Reward weights
        progress_reward_weight: float = 100.0,
        waypoint_bonus: float = 50.0,
        height_reward_weight: float = 2.0,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        healthy_reward: float = 5.0,
        
        # Health constraints
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple = (0.8, 3.0),
        check_healthy_z_relative: bool = False,
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        """
        Args:
            waypoints: List of (x, y) coordinates to visit in order
            waypoint_reach_threshold: Distance within which waypoint is considered reached
            stairs: List of stair configurations (start_x, num_steps, step_height, step_depth)
            terrain_width: Width of the terrain corridor
            progress_reward_weight: Weight for progress toward current waypoint
            waypoint_bonus: Large bonus for reaching each waypoint
            height_reward_weight: Weight for height gain (climbing stairs)
            forward_reward_weight: Weight for forward velocity
            ctrl_cost_weight: Weight for control cost
            contact_cost_weight: Weight for contact forces penalty
            healthy_reward: Reward for staying healthy
            terminate_when_unhealthy: Whether to end episode on fall
            healthy_z_range: Valid z-position range
            check_healthy_z_relative: If True, health check compares z to terrain height
            reset_noise_scale: Noise scale for initial state
            exclude_current_positions_from_observation: Whether to exclude global x,y from obs
        """
        utils.EzPickle.__init__(
            self,
            waypoints,
            waypoint_reach_threshold,
            stairs,
            terrain_width,
            progress_reward_weight,
            waypoint_bonus,
            height_reward_weight,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            check_healthy_z_relative,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        # Store configuration
        self._waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]
        self._waypoint_reach_threshold = waypoint_reach_threshold
        self._stairs = stairs
        self._terrain_width = terrain_width
        
        # Reward weights
        self._progress_reward_weight = progress_reward_weight
        self._waypoint_bonus = waypoint_bonus
        self._height_reward_weight = height_reward_weight
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        
        # Health and reset
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._check_healthy_z_relative = check_healthy_z_relative
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        # Track waypoint progress
        self._current_waypoint_index = 0
        self._waypoints_reached = 0
        self._prev_distance_to_waypoint = 0.0

        # Height grid for terrain perception
        self._height_grid_size = 5
        self._height_sample_distance = 0.3
        
        # Generate XML
        xml_content = self._generate_xml()
        self._temp_xml = tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', delete=False, encoding='utf-8'
        )
        self._temp_xml.write(xml_content)
        self._temp_xml.close()
        
        # Observation space: 376 (base) + 25 (height grid) + 2 (waypoint vector) + 1 (waypoint progress)
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(404,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            self._temp_xml.name,
            5,  # frame_skip
            observation_space=observation_space,
            **kwargs,
        )

    def __del__(self):
        """Clean up temporary XML file."""
        if hasattr(self, '_temp_xml'):
            try:
                os.unlink(self._temp_xml.name)
            except:
                pass

    def _generate_xml(self) -> str:
        """Generate MuJoCo XML with waypoints and stairs."""
        
        # Calculate terrain bounds
        all_x_coords = [wp[0] for wp in self._waypoints] + [0.0]
        if self._stairs:
            for start_x, num_steps, step_height, step_depth in self._stairs:
                all_x_coords.append(start_x + num_steps * step_depth)
        
        max_x = max(all_x_coords) + 10.0
        
        xml = '''<mujoco model="humanoid_circuit">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003"/>
    <size nkey="5" nuser_geom="1"/>
    <visual>
        <map fogend="5" fogstart="3"/>
    </visual>
    <asset>
        <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
'''
        
        # Add base terrain (flat ground)
        half_width = self._terrain_width / 2
        xml += f'''
        <!-- Base flat terrain -->
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" 
              pos="{max_x/2} 0 -0.1" rgba="0.8 0.9 0.8 1" 
              size="{max_x/2} {half_width} 0.1" type="box"/>
'''
        
        # Add stairs and elevated platforms
        for stair_idx, (start_x, num_steps, step_height, step_depth) in enumerate(self._stairs):
            # Add each step
            for i in range(num_steps):
                center_x = start_x + (i + 0.5) * step_depth
                height = (i + 0.5) * step_height
                half_depth = step_depth / 2
                half_height = step_height / 2
                
                xml += f'''
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="stair_{stair_idx}_{i+1}" 
              pos="{center_x} 0 {height}" rgba="0.7 0.8 0.7 1" 
              size="{half_depth} {half_width} {half_height}" type="box"/>
'''
            
            # Add elevated platform after stairs (extends to next stair section or end)
            stairs_end_x = start_x + num_steps * step_depth
            top_height = num_steps * step_height
            
            # Determine platform end (next stair start or max_x)
            if stair_idx + 1 < len(self._stairs):
                platform_end_x = self._stairs[stair_idx + 1][0]
            else:
                platform_end_x = max_x
            
            platform_length = platform_end_x - stairs_end_x
            if platform_length > 0.1:  # Only add if meaningful length
                platform_center_x = stairs_end_x + platform_length / 2
                platform_size_x = platform_length / 2
                
                xml += f'''
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="platform_{stair_idx}" 
              pos="{platform_center_x} 0 {top_height}" rgba="0.75 0.85 0.75 1" 
              size="{platform_size_x} {half_width} 0.05" type="box"/>
'''
        
        # Add waypoint markers (visual only)
        for wp_idx, (wp_x, wp_y) in enumerate(self._waypoints):
            xml += f'''
        <!-- Waypoint {wp_idx + 1} marker -->
        <geom name="waypoint_{wp_idx}" pos="{wp_x} {wp_y} 0.5" rgba="1 0 0 0.3" 
              size="0.3 0.3 0.5" type="cylinder" contype="0" conaffinity="0"/>
'''
        
        # Add humanoid body (standard model)
        xml += '''
        <body name="torso" pos="0 0 1.4">
            <camera name="track" mode="trackcom" pos="0 -4 0" xyaxes="1 0 0 0 0 1"/>
            <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
            <geom fromto="0 -.07 0 0 .07 0" name="torso1" size="0.07" type="capsule"/>
            <geom name="head" pos="0 0 .19" size=".09" type="sphere" user="258"/>
            <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size="0.06" type="capsule"/>
            <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
                <geom fromto="0 -.06 0 0 .06 0" name="lwaist" size="0.06" type="capsule"/>
                <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
                <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
                <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
                    <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
                    <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size="0.09" type="capsule"/>
                    <body name="right_thigh" pos="0 -0.1 -0.04">
                        <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.0080" axis="0 1 0" damping="5" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size="0.06" type="capsule"/>
                        <body name="right_shin" pos="0 0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size="0.049" type="capsule"/>
                            <body name="right_foot" pos="0 0 -0.45">
                                <geom name="right_foot" pos="0 0 0.1" size="0.075" type="sphere" user="0"/>
                            </body>
                        </body>
                    </body>
                    <body name="left_thigh" pos="0 0.1 -0.04">
                        <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule"/>
                        <body name="left_shin" pos="0 -0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule"/>
                            <body name="left_foot" pos="0 0 -0.45">
                                <geom name="left_foot" type="sphere" size="0.075" pos="0 0 0.1" user="0" />
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="right_upper_arm" pos="0 -0.17 0.06">
                <joint armature="0.0068" axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size="0.04 0.16" type="capsule"/>
                <body name="right_lower_arm" pos=".18 -.18 -.18">
                    <joint armature="0.0028" axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm" size="0.031" type="capsule"/>
                    <geom name="right_hand" pos=".18 .18 .18" size="0.04" type="sphere"/>
                </body>
            </body>
            <body name="left_upper_arm" pos="0 0.17 0.06">
                <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size="0.04 0.16" type="capsule"/>
                <body name="left_lower_arm" pos=".18 .18 -.18">
                    <joint armature="0.0028" axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm" size="0.031" type="capsule"/>
                    <geom name="left_hand" pos=".18 -.18 .18" size="0.04" type="sphere"/>
                </body>
            </body>
        </body>
    </worldbody>
    <tendon>
        <fixed name="left_hipknee">
            <joint coef="-1" joint="left_hip_y"/>
            <joint coef="1" joint="left_knee"/>
        </fixed>
        <fixed name="right_hipknee">
            <joint coef="-1" joint="right_hip_y"/>
            <joint coef="1" joint="right_knee"/>
        </fixed>
    </tendon>
    <actuator>
        <motor gear="100" joint="abdomen_y" name="abdomen_y"/>
        <motor gear="100" joint="abdomen_z" name="abdomen_z"/>
        <motor gear="100" joint="abdomen_x" name="abdomen_x"/>
        <motor gear="100" joint="right_hip_x" name="right_hip_x"/>
        <motor gear="100" joint="right_hip_z" name="right_hip_z"/>
        <motor gear="300" joint="right_hip_y" name="right_hip_y"/>
        <motor gear="200" joint="right_knee" name="right_knee"/>
        <motor gear="100" joint="left_hip_x" name="left_hip_x"/>
        <motor gear="100" joint="left_hip_z" name="left_hip_z"/>
        <motor gear="300" joint="left_hip_y" name="left_hip_y"/>
        <motor gear="200" joint="left_knee" name="left_knee"/>
        <motor gear="25" joint="right_shoulder1" name="right_shoulder1"/>
        <motor gear="25" joint="right_shoulder2" name="right_shoulder2"/>
        <motor gear="25" joint="right_elbow" name="right_elbow"/>
        <motor gear="25" joint="left_shoulder1" name="left_shoulder1"/>
        <motor gear="25" joint="left_shoulder2" name="left_shoulder2"/>
        <motor gear="25" joint="left_elbow" name="left_elbow"/>
    </actuator>
</mujoco>'''
        
        return xml

    def _get_terrain_height_at(self, x, y):
        """Get terrain height at position based on stairs configuration."""
        height = 0.0
        
        # Check each stair section
        for start_x, num_steps, step_height, step_depth in self._stairs:
            end_x = start_x + num_steps * step_depth
            
            if start_x <= x < end_x:
                # On stairs
                step_index = int((x - start_x) / step_depth)
                step_index = min(step_index, num_steps - 1)
                height = max(height, step_index * step_height)
            elif x >= end_x:
                # After stairs
                total_height = num_steps * step_height
                height = max(height, total_height)
        
        return height

    def _get_height_grid(self):
        """Sample terrain heights in 5x5 grid around agent."""
        current_x = self.data.qpos[0]
        current_y = self.data.qpos[1]
        current_z = self.data.qpos[2]
        
        heights = []
        grid_half_size = (self._height_grid_size - 1) * self._height_sample_distance / 2
        
        for i in range(self._height_grid_size):
            for j in range(self._height_grid_size):
                dx = -grid_half_size + i * self._height_sample_distance
                dy = -grid_half_size + j * self._height_sample_distance
                
                sample_x = current_x + dx
                sample_y = current_y + dy
                
                terrain_height = self._get_terrain_height_at(sample_x, sample_y)
                relative_height = terrain_height - current_z
                heights.append(relative_height)
        
        return np.array(heights, dtype=np.float64)

    @property
    def current_waypoint(self):
        """Get current target waypoint."""
        if self._current_waypoint_index < len(self._waypoints):
            return self._waypoints[self._current_waypoint_index]
        else:
            # All waypoints reached
            return self._waypoints[-1]

    @property
    def healthy_reward(self):
        return float(self.is_healthy) * self._healthy_reward

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def contact_cost(self):
        contact_forces = self.data.cfrc_ext
        return self._contact_cost_weight * np.sum(np.square(contact_forces))

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        current_z = self.data.qpos[2]
        
        if self._check_healthy_z_relative:
            # Check height relative to terrain
            terrain_h = self._get_terrain_height_at(self.data.qpos[0], self.data.qpos[1])
            relative_z = current_z - terrain_h
            is_healthy = min_z < relative_z < max_z
        else:
            # Check absolute height
            is_healthy = min_z < current_z < max_z
            
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
        
        # Height grid for terrain perception
        height_grid = self._get_height_grid()
        
        # Relative vector to current waypoint
        current_xy = position[0:2]
        vector_to_waypoint = self.current_waypoint - current_xy
        
        # Waypoint progress (normalized)
        waypoint_progress = np.array([self._waypoints_reached / len(self._waypoints)], dtype=np.float64)

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
                height_grid,
                vector_to_waypoint,
                waypoint_progress,
            )
        )

    def step(self, action):
        prev_xy_position = self.data.qpos[0:2].copy()
        prev_z_position = self.data.qpos[2]
        prev_dist = np.linalg.norm(self.current_waypoint - prev_xy_position)

        self.do_simulation(action, self.frame_skip)

        xy_position = self.data.qpos[0:2].copy()
        z_position = self.data.qpos[2]
        xy_velocity = self.data.qvel[0:2].copy()
        
        # Current distance to waypoint
        dist = np.linalg.norm(self.current_waypoint - xy_position)

        # Progress reward (getting closer to current waypoint)
        progress_reward = (prev_dist - dist) * self._progress_reward_weight

        # Forward reward (general forward movement)
        forward_reward = self._forward_reward_weight * xy_velocity[0]

        # Height reward (climbing stairs)
        height_gained = z_position - prev_z_position
        height_reward = self._height_reward_weight * max(0, height_gained)

        # Waypoint bonus
        waypoint_reward = 0.0
        waypoint_reached = False
        if dist < self._waypoint_reach_threshold:
            if self._current_waypoint_index < len(self._waypoints):
                waypoint_reward = self._waypoint_bonus
                self._current_waypoint_index += 1
                self._waypoints_reached += 1
                waypoint_reached = True

        # Circuit completion bonus
        circuit_complete = (self._waypoints_reached == len(self._waypoints))
        
        # Healthy reward
        healthy_reward = self.healthy_reward
        
        # Costs
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        # Total reward
        reward = (
            progress_reward + 
            forward_reward + 
            height_reward + 
            waypoint_reward + 
            healthy_reward - 
            ctrl_cost - 
            contact_cost
        )

        observation = self._get_obs()
        terminated = self.terminated

        info = {
            "reward_progress": progress_reward,
            "reward_forward": forward_reward,
            "reward_height": height_reward,
            "reward_waypoint": waypoint_reward,
            "reward_survive": healthy_reward,
            "cost_ctrl": ctrl_cost,
            "cost_contact": contact_cost,
            "x_position": xy_position[0],
            "y_position": xy_position[1],
            "z_position": z_position,
            "distance_to_waypoint": dist,
            "current_waypoint_index": self._current_waypoint_index,
            "waypoints_reached": self._waypoints_reached,
            "circuit_complete": circuit_complete,
            "waypoint_just_reached": waypoint_reached,
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

        qpos[0] = 0.0  # x position
        qpos[1] = 0.0  # y position
        qpos[2] = 1.4  # z position

        self.set_state(qpos, qvel)
        
        # Reset waypoint tracking
        self._current_waypoint_index = 0
        self._waypoints_reached = 0
        self._prev_distance_to_waypoint = np.linalg.norm(self.current_waypoint - qpos[0:2])

        observation = self._get_obs()
        return observation
