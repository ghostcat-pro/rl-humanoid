from __future__ import annotations
import os
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

def maybe_load_vecnormalize(venv, vecnorm_path: str | None):
    """
    If a VecNormalize .pkl exists at `vecnorm_path`, load it into the given `venv`.
    Switches to eval mode (no running-stats updates, no reward normalization).
    Otherwise, returns `venv` unchanged.

    Parameters
    ----------
    venv : VecEnv
        A vectorized env (can already be wrapped by VecNormalize or be a base vec env).
    vecnorm_path : str | None
        Path to the saved VecNormalize stats (created via venv.save(...)).

    Returns
    -------
    venv : VecEnv
        The same `venv` instance with VecNormalize stats loaded (if file exists), or unchanged.
    """
    if vecnorm_path and os.path.isfile(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    return venv
