"""Configurable Humanoid Stairs Environment with flexible parameters."""

from typing import Dict, Any, Optional, Literal
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import tempfile
import xml.etree.ElementTree as ET


class HumanoidStairsConfigurableEnv(MujocoEnv, utils.EzPickle):
    """
    Highly configurable humanoid stairs environment.
    
    Allows customization of:
    - Flat distance before stairs
    - Number of steps
    - Step dimensions (height, depth/length)
    - End platform configuration
    - Stairs going down after reaching top
    - Abyss ending (no end platform)
    
    This provides maximum flexibility for curriculum learning and experimentation.
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
        # Terrain configuration
        flat_distance_before_stairs: float = 20.0,
        num_steps: int = 10,
        step_height: float = 0.15,
        step_depth: float = 0.6,
        end_platform_length: float = 5.0,
        stairs_after_top: bool = False,
        num_steps_down: int = 5,
        end_with_abyss: bool = False,
        # Reward configuration
        forward_reward_weight: float = 1.25,
        height_reward_weight: float = 2.0,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        healthy_reward: float = 5.0,
        step_bonus: float = 10.0,
        # Health constraints
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple = (0.8, 3.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        """
        Args:
            flat_distance_before_stairs: Distance of flat terrain before first step (meters)
            num_steps: Number of stairs going up
            step_height: Height of each step (meters)
            step_depth: Depth/length of each step (meters)
            end_platform_length: Length of platform at top (meters, 0 if end_with_abyss=True)
            stairs_after_top: If True, stairs continue downward after top platform
            num_steps_down: Number of steps going down (only if stairs_after_top=True)
            end_with_abyss: If True, no end platform (agent must stop at top step)
            forward_reward_weight: Weight for forward progress reward
            height_reward_weight: Weight for height gain reward
            ctrl_cost_weight: Weight for control cost penalty
            contact_cost_weight: Weight for contact force penalty (encourages smooth movement)
            healthy_reward: Reward for staying upright/alive
            step_bonus: Bonus reward for reaching a new step
            terminate_when_unhealthy: Whether to terminate episode when agent falls
            healthy_z_range: Valid z-position range for agent
            reset_noise_scale: Noise scale for initial state
            exclude_current_positions_from_observation: Whether to exclude global x,y from obs
        """
        utils.EzPickle.__init__(
            self,
            flat_distance_before_stairs,
            num_steps,
            step_height,
            step_depth,
            end_platform_length,
            stairs_after_top,
            num_steps_down,
            end_with_abyss,
            forward_reward_weight,
            height_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            step_bonus,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        # Store configuration
        self._flat_distance = flat_distance_before_stairs
        self._num_steps = num_steps
        self._step_height = step_height
        self._step_depth = step_depth
        self._end_platform_length = end_platform_length if not end_with_abyss else 0.0
        self._stairs_after_top = stairs_after_top
        self._num_steps_down = num_steps_down if stairs_after_top else 0
        self._end_with_abyss = end_with_abyss
        
        # Reward weights
        self._forward_reward_weight = forward_reward_weight
        self._height_reward_weight = height_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._step_bonus = step_bonus
        
        # Health and reset
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # Calculate stair boundaries
        self._stairs_start_x = flat_distance_before_stairs
        self._stairs_end_x = self._stairs_start_x + (num_steps * step_depth)
        self._max_height = num_steps * step_height
        self._steps_x_bounds = []
        
        for i in range(num_steps):
            start_x = self._stairs_start_x + i * step_depth
            end_x = start_x + step_depth
            self._steps_x_bounds.append((start_x, end_x))
        
        # Calculate downward stairs if applicable
        self._steps_down_x_bounds = []
        if stairs_after_top:
            down_start_x = self._stairs_end_x + end_platform_length
            for i in range(num_steps_down):
                start_x = down_start_x + i * step_depth
                end_x = start_x + step_depth
                self._steps_down_x_bounds.append((start_x, end_x))
        
        # Track progress
        self._max_step_reached = 0

        # Generate XML dynamically based on configuration
        xml_content = self._generate_xml()
        
        # Save to temporary file
        self._temp_xml = tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', delete=False, encoding='utf-8'
        )
        self._temp_xml.write(xml_content)
        self._temp_xml.close()
        
        # Terrain perception: 5x5 grid around agent = 25 height values
        self._height_grid_size = 5
        self._height_sample_distance = 0.3
        height_grid_dims = self._height_grid_size * self._height_grid_size
        
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376 + height_grid_dims,), dtype=np.float64
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
        """Generate MuJoCo XML based on configuration parameters."""
        
        # Start with base humanoid XML (we'll build it programmatically)
        xml_template = '''<mujoco model="humanoid_configurable">
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
        
        # Add starting platform
        platform_center_x = self._flat_distance / 2
        platform_size_x = self._flat_distance / 2
        xml_template += f'''
        <!-- Starting platform: {self._flat_distance}m of flat terrain -->
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="start_platform" 
              pos="{platform_center_x} 0 0" rgba="0.8 0.9 0.8 1" 
              size="{platform_size_x} 2 0.125" type="box"/>
'''
        
        # Add stairs going up
        for i in range(self._num_steps):
            center_x = self._stairs_start_x + (i + 0.5) * self._step_depth
            height = (i + 0.5) * self._step_height
            half_depth = self._step_depth / 2
            half_height = self._step_height / 2
            
            xml_template += f'''
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="stair_up_{i+1}" 
              pos="{center_x} 0 {height}" rgba="0.7 0.8 0.7 1" 
              size="{half_depth} 2 {half_height}" type="box"/>
'''
        
        # Add end platform (if not abyss)
        if not self._end_with_abyss and self._end_platform_length > 0:
            platform_center_x = self._stairs_end_x + self._end_platform_length / 2
            platform_size_x = self._end_platform_length / 2
            platform_height = self._max_height
            
            xml_template += f'''
        <!-- End platform at top -->
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="end_platform" 
              pos="{platform_center_x} 0 {platform_height}" rgba="0.8 0.9 0.8 1" 
              size="{platform_size_x} 2 0.125" type="box"/>
'''
        
        # Add stairs going down (if configured)
        if self._stairs_after_top:
            down_start_x = self._stairs_end_x + self._end_platform_length
            for i in range(self._num_steps_down):
                center_x = down_start_x + (i + 0.5) * self._step_depth
                # Height decreases as we go down
                height = self._max_height - ((i + 0.5) * self._step_height)
                half_depth = self._step_depth / 2
                half_height = self._step_height / 2
                
                xml_template += f'''
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="stair_down_{i+1}" 
              pos="{center_x} 0 {height}" rgba="0.7 0.7 0.8 1" 
              size="{half_depth} 2 {half_height}" type="box"/>
'''
        
        # Add humanoid body (standard humanoid model)
        xml_template += '''
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
                    <camera pos="0 0 0"/>
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
        
        return xml_template

    def _get_terrain_height_at(self, x, y):
        """Get terrain height at position based on configuration."""
        # Starting platform
        if x < self._stairs_start_x:
            return 0.0
        
        # Stairs going up
        elif x < self._stairs_end_x:
            step_index = int((x - self._stairs_start_x) / self._step_depth)
            step_index = min(step_index, self._num_steps - 1)
            return step_index * self._step_height
        
        # End platform
        elif x < self._stairs_end_x + self._end_platform_length:
            return self._max_height
        
        # Stairs going down (if configured)
        elif self._stairs_after_top:
            down_start_x = self._stairs_end_x + self._end_platform_length
            down_end_x = down_start_x + (self._num_steps_down * self._step_depth)
            
            if x < down_end_x:
                step_index = int((x - down_start_x) / self._step_depth)
                step_index = min(step_index, self._num_steps_down - 1)
                return self._max_height - (step_index * self._step_height)
            else:
                # After downward stairs
                return self._max_height - (self._num_steps_down * self._step_height)
        
        # Abyss or beyond
        elif self._end_with_abyss:
            return self._max_height  # Agent should not go beyond
        
        else:
            return self._max_height

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
                height_grid,
            )
        )

    def step(self, action):
        prev_xy_position = self.data.qpos[0:2].copy()
        prev_z_position = self.data.qpos[2]

        self.do_simulation(action, self.frame_skip)

        xy_position = self.data.qpos[0:2].copy()
        z_position = self.data.qpos[2]
        xy_velocity = self.data.qvel[0:2].copy()

        # Forward reward
        forward_reward = self._forward_reward_weight * xy_velocity[0]

        # Height reward
        height_gained = z_position - prev_z_position
        height_reward = self._height_reward_weight * max(0, height_gained)

        # Healthy reward
        healthy_reward = self.healthy_reward
        
        # Step reward
        step_reward = 0.0
        current_step = 0
        for i, (start_x, end_x) in enumerate(self._steps_x_bounds):
            if xy_position[0] >= start_x:
                current_step = i + 1
        
        if current_step > self._max_step_reached:
            step_reward = self._step_bonus * (current_step - self._max_step_reached)
            self._max_step_reached = current_step

        # Control cost
        ctrl_cost = self.control_cost(action)
        
        # Contact cost
        contact_cost = self.contact_cost

        # Total reward
        reward = forward_reward + height_reward + healthy_reward + step_reward - ctrl_cost - contact_cost

        observation = self._get_obs()
        terminated = self.terminated

        info = {
            "reward_forward": forward_reward,
            "reward_height": height_reward,
            "reward_survive": healthy_reward,
            "reward_step": step_reward,
            "cost_ctrl": ctrl_cost,
            "cost_contact": contact_cost,
            "x_position": xy_position[0],
            "y_position": xy_position[1],
            "z_position": z_position,
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

        qpos[0] = 0.0  # x position
        qpos[1] = 0.0  # y position
        qpos[2] = 1.4  # z position

        self.set_state(qpos, qvel)
        self._max_step_reached = 0

        observation = self._get_obs()
        return observation
