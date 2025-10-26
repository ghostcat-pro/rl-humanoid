# ============================================
# RL-Humanoid Project Setup Script (Windows)
# ============================================
# Run this script from inside the rl-humanoid directory
# Example:
# PS C:\projects\rl-humanoid> .\setup_rl_humanoid.ps1

# Create folder structure
New-Item -ItemType Directory -Force -Path conf, conf\env, conf\algo, conf\training, utils | Out-Null

# -------------------------------
# requirements.txt
# -------------------------------
@"
gymnasium[mujoco]>=0.29
mujoco>=3.0.0
stable-baselines3>=2.3.0
hydra-core>=1.3.2
omegaconf>=2.3.0
tensorboard>=2.16
numpy>=1.24
"@ | Out-File -Encoding UTF8 requirements.txt

# -------------------------------
# conf/main.yaml
# -------------------------------
@"
defaults:
  - env: humanoid
  - algo: ppo
  - training: default

exp_name: "rl-humanoid"
seed: 42

paths:
  log_root: \${hydra:runtime.output_dir}

vecnorm:
  enabled: true
  clip_obs: 10.0
  gamma: 0.99
  epsilon: 1e-8
"@ | Out-File -Encoding UTF8 conf\main.yaml

# -------------------------------
# conf/env/humanoid.yaml
# -------------------------------
@"
name: Humanoid-v5
make_kwargs:
  render_mode: null
vec_env:
  n_envs: 8
  start_method: spawn
  monitor: true
"@ | Out-File -Encoding UTF8 conf\env\humanoid.yaml

# -------------------------------
# conf/env/walker2d.yaml
# -------------------------------
@"
name: Walker2d-v5
make_kwargs:
  render_mode: null
vec_env:
  n_envs: 8
  start_method: spawn
  monitor: true
"@ | Out-File -Encoding UTF8 conf\env\walker2d.yaml

# -------------------------------
# conf/algo/ppo.yaml
# -------------------------------
@"
policy: MlpPolicy
hyperparams:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 8192
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
"@ | Out-File -Encoding UTF8 conf\algo\ppo.yaml

# -------------------------------
# conf/training/default.yaml
# -------------------------------
@"
total_timesteps: 1_000_000
log_interval: 10
checkpoint_every_steps: 250_000
"@ | Out-File -Encoding UTF8 conf\training\default.yaml

# -------------------------------
# conf/training/long.yaml
# -------------------------------
@"
total_timesteps: 5_000_000
log_interval: 10
checkpoint_every_steps: 500_000
"@ | Out-File -Encoding UTF8 conf\training\long.yaml

# -------------------------------
# utils/make_env.py
# -------------------------------
@"
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def make_single_env(env_id, make_kwargs, monitor=True, seed=None):
    def _init():
        env = gym.make(env_id, **(make_kwargs or {}))
        if seed is not None:
            env.reset(seed=seed)
        if monitor:
            env = Monitor(env)
        return env
    return _init

def make_vector_env(env_id, n_envs=8, start_method="spawn", make_kwargs=None,
                    monitor=True, seed=None, vecnormalize_kwargs=None, use_subproc=True):
    make_kwargs = make_kwargs or {}
    env_fns = [make_single_env(env_id, make_kwargs, monitor, None if seed is None else seed + i)
               for i in range(n_envs)]

    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    venv = vec_cls(env_fns, start_method=start_method if vec_cls is SubprocVecEnv else None)

    if vecnormalize_kwargs:
        venv = VecNormalize(venv, **vecnormalize_kwargs)
    return venv
"@ | Out-File -Encoding UTF8 utils\make_env.py

# -------------------------------
# utils/callbacks.py
# -------------------------------
@"
import os
from stable_baselines3.common.callbacks import BaseCallback

class CheckpointAndVecNormCallback(BaseCallback):
    def __init__(self, save_dir, save_freq_steps, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq_steps = save_freq_steps
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps > 0 and (self.num_timesteps % self.save_freq_steps == 0):
            model_path = os.path.join(self.save_dir, f"model_{self.num_timesteps}.zip")
            self.model.save(model_path)
            try:
                venv = self.model.get_env()
                if hasattr(venv, "save"):
                    venv.save(os.path.join(self.save_dir, f"vecnormalize_{self.num_timesteps}.pkl"))
            except Exception as e:
                print(f"[WARN] VecNormalize save failed: {e}")
        return True
"@ | Out-File -Encoding UTF8 utils\callbacks.py

# -------------------------------
# utils/vecnorm_io.py
# -------------------------------
@"
import os
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

def maybe_load_vecnormalize(venv, vecnorm_path=None):
    if vecnorm_path and os.path.isfile(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    return venv
"@ | Out-File -Encoding UTF8 utils\vecnorm_io.py

# -------------------------------
# train.py
# -------------------------------
@"
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from utils.make_env import make_vector_env
from utils.callbacks import CheckpointAndVecNormCallback

@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    print("\\n=== Merged Config ===\\n", OmegaConf.to_yaml(cfg), flush=True)

    set_random_seed(cfg.seed)

    vecnorm_kwargs = None
    if cfg.get("vecnorm") and cfg.vecnorm.enabled:
        vecnorm_kwargs = dict(
            norm_obs=True,
            norm_reward=True,
            clip_obs=cfg.vecnorm.clip_obs,
            gamma=cfg.vecnorm.gamma,
            epsilon=cfg.vecnorm.epsilon,
        )

    venv = make_vector_env(
        env_id=cfg.env.name,
        n_envs=cfg.env.vec_env.n_envs,
        start_method=cfg.env.vec_env.start_method,
        make_kwargs=cfg.env.make_kwargs,
        monitor=cfg.env.vec_env.monitor,
        seed=cfg.seed,
        vecnormalize_kwargs=vecnorm_kwargs,
        use_subproc=True,
    )

    run_dir = cfg.paths.log_root
    logger = configure(run_dir, ["stdout", "tensorboard", "csv"])

    hp = cfg.algo.hyperparams
    n_steps = int(hp.n_steps)
    n_envs = int(cfg.env.vec_env.n_envs)
    assert (n_envs * n_steps) % int(hp.batch_size) == 0, \
        f"batch_size must divide n_envs*n_steps ({n_envs*n_steps})."

    model = PPO(
        policy=cfg.algo.policy,
        env=venv,
        learning_rate=hp.learning_rate,
        n_steps=n_steps,
        batch_size=int(hp.batch_size),
        n_epochs=int(hp.n_epochs),
        gamma=hp.gamma,
        gae_lambda=hp.gae_lambda,
        clip_range=hp.clip_range,
        ent_coef=hp.ent_coef,
        vf_coef=hp.vf_coef,
        max_grad_norm=hp.max_grad_norm,
        verbose=1,
    )
    model.set_logger(logger)

    ckpt_cb = CheckpointAndVecNormCallback(
        save_dir=os.path.join(run_dir, "checkpoints"),
        save_freq_steps=int(cfg.training.checkpoint_every_steps),
        verbose=1,
    )

    model.learn(total_timesteps=int(cfg.training.total_timesteps),
                callback=ckpt_cb, log_interval=cfg.training.log_interval)

    model.save(os.path.join(run_dir, "final_model.zip"))
    try:
        venv.save(os.path.join(run_dir, "vecnormalize_final.pkl"))
    except Exception as e:
        print(f"[WARN] VecNormalize final save failed: {e}")

    print(f"\\nRun dir: {run_dir}\\nSaved final model and logs.")

if __name__ == "__main__":
    main()
"@ | Out-File -Encoding UTF8 train.py

# -------------------------------
# evaluate.py
# -------------------------------
@"
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.make_env import make_single_env
from utils.vecnorm_io import maybe_load_vecnormalize

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="Humanoid-v5")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--vecnorm_path", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    make_kwargs = {"render_mode": "human"} if args.render else {"render_mode": None}
    env_fn = make_single_env(args.env_id, make_kwargs, monitor=False, seed=123)
    venv = DummyVecEnv([env_fn])
    if args.vecnorm_path:
        venv = maybe_load_vecnormalize(venv, args.vecnorm_path)
    model = PPO.load(args.model_path)
    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_rew += float(reward)
        print(f"Episode {ep+1}: reward={ep_rew:.2f}")

if __name__ == "__main__":
    main()
"@ | Out-File -Encoding UTF8 evaluate.py

# -------------------------------
# .gitignore
# -------------------------------
@"
__pycache__/
*.pyc
.venv/
outputs/
multirun/
*/checkpoints/
*.zip
*.pkl
.DS_Store
"@ | Out-File -Encoding UTF8 .gitignore

# -------------------------------
# README.md
# -------------------------------
@"
# RL Humanoid — SB3 + Hydra Skeleton
"@ | Out-File -Encoding UTF8 README.md
## Install
