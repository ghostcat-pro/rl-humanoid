from __future__ import annotations
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

#added line
from utils.reward_wrappers import UprightAndEffortWrapper

def make_single_env(env_id: str, make_kwargs: dict, monitor: bool = True, seed: int | None = None):
    """Factory returning a thunk to create one env instance."""
    def _init():
        # Remove custom wrapper keys before passing to gym.make
        env_kwargs = dict(make_kwargs or {})
        upright_w = env_kwargs.pop('upright_weight', 0.5)
        effort_w = env_kwargs.pop('effort_weight', 0.001)
        lateral_w = env_kwargs.pop('lateral_penalty_weight', 1.0)
        env = gym.make(env_id, **env_kwargs)
        if seed is not None:
            env.reset(seed=seed)
        env = UprightAndEffortWrapper(env, upright_w=upright_w, effort_w=effort_w, lateral_w=lateral_w)
        if monitor:
            env = Monitor(env)
        return env
    return _init

def make_vector_env(
    env_id: str,
    n_envs: int = 8,
    start_method: str = "spawn",
    make_kwargs: dict | None = None,
    monitor: bool = True,
    seed: int | None = None,
    vecnormalize_kwargs: dict | None = None,
    use_subproc: bool = True,
):
    """Create a vectorized env and (optionally) wrap with VecNormalize."""
    make_kwargs = make_kwargs or {}
    env_fns = [
        make_single_env(env_id, make_kwargs, monitor, None if seed is None else seed + i)
        for i in range(n_envs)
    ]

    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    if vec_cls is SubprocVecEnv:
        venv = vec_cls(env_fns, start_method=start_method)
    else:
        venv = vec_cls(env_fns)

    if vecnormalize_kwargs:
        venv = VecNormalize(venv, **vecnormalize_kwargs)
    return venv

