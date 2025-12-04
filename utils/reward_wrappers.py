import gymnasium as gym
import numpy as np

class UprightAndEffortWrapper(gym.Wrapper):
    """
    Small shaping to stabilize gait:
    + upright bonus (torso alignment)
    - action L2 penalty (discourage frantic torques)
    """
    def __init__(self, env, upright_w=0.05, effort_w=0.001, lateral_w=1.0):
        super().__init__(env)
        self.upright_w = float(upright_w)
        self.effort_w = float(effort_w)
        self.lateral_w = float(lateral_w)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Penalize effort
        reward -= self.effort_w * float(np.sum(np.square(action)))

        # Upright bonus via torso quaternion (MuJoCo)
        try:
            torso_id = self.env.unwrapped.model.body_name2id("torso")
            w, x, y, z = self.env.unwrapped.data.xquat[torso_id]  # [w, x, y, z]
            # Heuristic “uprightness”: prefer pitch/roll near 0
            upright = (w*w - x*x - y*y + z*z)
            reward += self.upright_w * max(0.0, float(upright))
        except Exception:
            pass

        # Penalize lateral deviation (y-axis)
        try:
            # MuJoCo: data.qpos[0:3] = [x, y, z] of root body
            y_pos = float(self.env.unwrapped.data.qpos[1])
            reward -= self.lateral_w * abs(y_pos)
        except Exception:
            pass
        return obs, reward, terminated, truncated, info
