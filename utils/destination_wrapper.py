"""Wrapper to add destination/target navigation to any Humanoid-like environment."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class DestinationWrapper(gym.Wrapper):
    """
    Wrapper that adds target destination navigation to a humanoid environment.
    
    Adds:
    - 2D relative vector to target in observation space
    - Distance-based reward shaping
    - Reach bonus when close to target
    
    This wrapper can be applied to any environment with accessible qpos data.
    """
    
    def __init__(
        self,
        env,
        target_position=(5.0, 0.0),
        distance_reward_weight=1.0,
        reach_reward_weight=10.0,
        reach_threshold=0.5,
    ):
        """
        Args:
            env: Base gymnasium environment
            target_position: (x, y) coordinates of target
            distance_reward_weight: Weight for distance improvement reward
            reach_reward_weight: Bonus reward when within reach_threshold
            reach_threshold: Distance threshold for reach bonus (meters)
        """
        super().__init__(env)
        
        self.target_position = np.array(target_position, dtype=np.float64)
        self.distance_reward_weight = distance_reward_weight
        self.reach_reward_weight = reach_reward_weight
        self.reach_threshold = reach_threshold
        
        # Track previous distance for progress reward
        self.prev_distance = None
        
        # Extend observation space: add 2 dimensions for relative target position
        old_obs_space = env.observation_space
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(old_obs_space.shape[0] + 2,),
            dtype=np.float64
        )
    
    def _get_current_position(self):
        """Get current (x, y) position from environment."""
        return self.env.unwrapped.data.qpos[0:2].copy()
    
    def _calculate_vector_to_target(self):
        """Calculate relative vector from current position to target."""
        current_xy = self._get_current_position()
        return self.target_position - current_xy
    
    def _augment_observation(self, obs):
        """Add relative target position to observation."""
        vector_to_target = self._calculate_vector_to_target()
        return np.concatenate([obs, vector_to_target])
    
    def reset(self, **kwargs):
        """Reset environment and initialize distance tracking."""
        obs, info = self.env.reset(**kwargs)
        
        # Initialize distance tracking
        vector_to_target = self._calculate_vector_to_target()
        self.prev_distance = np.linalg.norm(vector_to_target)
        
        # Augment observation
        obs = self._augment_observation(obs)
        
        # Add target info to info dict
        info['target_position'] = self.target_position
        info['distance_to_target'] = self.prev_distance
        
        return obs, info
    
    def step(self, action):
        """Step environment and add destination-based rewards."""
        # Take step in base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate current distance to target
        vector_to_target = self._calculate_vector_to_target()
        current_distance = np.linalg.norm(vector_to_target)
        
        # Progress reward: positive if getting closer, negative if moving away
        if self.prev_distance is not None:
            progress = (self.prev_distance - current_distance)
            progress_reward = progress * self.distance_reward_weight * 100.0
        else:
            progress_reward = 0.0
        
        # Reach bonus: extra reward when close to target
        reach_reward = 0.0
        if current_distance < self.reach_threshold:
            reach_reward = self.reach_reward_weight
        
        # Add destination rewards to base reward
        reward += progress_reward + reach_reward
        
        # Update distance tracking
        self.prev_distance = current_distance
        
        # Augment observation
        obs = self._augment_observation(obs)
        
        # Add destination info to info dict
        info['reward_progress'] = progress_reward
        info['reward_reach'] = reach_reward
        info['distance_to_target'] = current_distance
        info['target_position'] = self.target_position
        current_pos = self._get_current_position()
        info['x_position'] = current_pos[0]
        info['y_position'] = current_pos[1]
        
        return obs, reward, terminated, truncated, info
