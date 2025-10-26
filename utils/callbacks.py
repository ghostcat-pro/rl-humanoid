from __future__ import annotations
import os
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointAndVecNormCallback(BaseCallback):
    """
    Save the SB3 model and (if present) VecNormalize statistics every `save_freq_steps`.

    - Artifacts are written under `save_dir/`:
        model_<steps>.zip
        vecnormalize_<steps>.pkl
    """
    def __init__(self, save_dir: str, save_freq_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq_steps = int(save_freq_steps)
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # num_timesteps is the global counter across learn()
        if self.num_timesteps > 0 and (self.num_timesteps % self.save_freq_steps == 0):
            model_path = os.path.join(self.save_dir, f"model_{self.num_timesteps}.zip")
            self.model.save(model_path)

            # Try saving VecNormalize stats (if env is wrapped)
            try:
                venv = self.model.get_env()
                if hasattr(venv, "save"):  # VecNormalize exposes .save()
                    venv.save(os.path.join(self.save_dir, f"vecnormalize_{self.num_timesteps}.pkl"))
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] VecNormalize save failed at {self.num_timesteps} steps: {e}")
        return True

