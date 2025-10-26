from __future__ import annotations
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def make_single_env(env_id: str, make_kwargs: dict, monitor: bool = True, seed: int | None = None):
    """Factory returning a thunk to create one env instance."""
    def _init():
        env = gym.make(env_id, **(make_kwargs or {}))
        if seed is not None:
            env.reset(seed=seed)
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
    venv = vec_cls(env_fns, start_method=start_method if vec_cls is SubprocVecEnv else None)

    if vecnormalize_kwargs:
        venv = VecNormalize(venv, **vecnormalize_kwargs)
    return venv

