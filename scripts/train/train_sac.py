"""
Train a SAC (Soft Actor-Critic) agent on Gymnasium MuJoCo environments.

Usage:
    python scripts/train/train_sac.py env=humanoid algo=sac
    python scripts/train/train_sac.py env=humanoid_stairs algo=sac training=long
    
Outputs are saved to outputs_sac/ directory (separate from PPO outputs).
"""
from __future__ import annotations
import os
import sys
import time
from datetime import timedelta
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.make_env import make_vector_env
from utils.callbacks import CheckpointAndVecNormCallback

from hydra.core.hydra_config import HydraConfig

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.make_env import make_single_env

# Import custom environments to register them
import envs


@hydra.main(config_path="../../conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """
    Train a SAC agent on a Gymnasium MuJoCo environment using Hydra configs.
    """
    print("\n=== Merged Config ===\n", OmegaConf.to_yaml(cfg), flush=True)

    # Set random seed
    set_random_seed(cfg.seed)

    # VecNormalize configuration
    vecnorm_kwargs = None
    if cfg.get("vecnorm") and cfg.vecnorm.enabled:
        vecnorm_kwargs = dict(
            norm_obs=True,
            norm_reward=True,
            clip_obs=cfg.vecnorm.clip_obs,
            gamma=cfg.vecnorm.gamma,
            epsilon=cfg.vecnorm.epsilon,
        )

    # SAC typically uses a single environment (off-policy doesn't benefit as much from parallel envs)
    # But we still support vectorized envs if configured
    n_envs = cfg.env.vec_env.get("n_envs", 1)

    # For SAC, we recommend using 1 environment, but allow more if explicitly configured
    if n_envs > 1:
        print(f"[INFO] Using {n_envs} parallel environments. Note: SAC is off-policy and typically uses 1 env.")

    # Create vectorized environment
    venv = make_vector_env(
        env_id=cfg.env.name,
        n_envs=n_envs,
        start_method=cfg.env.vec_env.start_method,
        make_kwargs=cfg.env.make_kwargs,
        monitor=cfg.env.vec_env.monitor,
        seed=cfg.seed,
        vecnormalize_kwargs=vecnorm_kwargs,
        use_subproc=(n_envs > 1),
    )

    # Configure logger (Hydra gives each run a unique output dir)
    run_dir = HydraConfig.get().runtime.output_dir
    logger = configure(run_dir, ["stdout", "tensorboard", "csv"])

    # === Evaluation environment and callback (save best model) ===
    eval_make_kwargs = dict(cfg.env.make_kwargs) if cfg.env.make_kwargs else {}
    eval_make_kwargs["render_mode"] = None
    eval_env = DummyVecEnv([make_single_env(cfg.env.name, eval_make_kwargs, monitor=False, seed=cfg.seed + 10)])

    # If training uses VecNormalize, mirror its stats into the eval env
    if vecnorm_kwargs:
        from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
        eval_env = VecNormalize(eval_env, **vecnorm_kwargs)
        if hasattr(venv, "obs_rms"):
            eval_env.obs_rms = venv.obs_rms
        if hasattr(venv, "ret_rms"):
            eval_env.ret_rms = venv.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False

    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=eval_dir,
        log_path=eval_dir,
        eval_freq=int(cfg.training.checkpoint_every_steps // 2),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # Get hyperparameters
    hp = cfg.algo.hyperparams

    # Instantiate or load SAC model
    resuming = cfg.get("resume_from") and cfg.resume_from
    if resuming:
        print(f"\n=== Resuming from checkpoint: {cfg.resume_from} ===")

        # Load VecNormalize stats if available
        vecnorm_path = cfg.resume_from.replace("model_", "vecnormalize_").replace(".zip", ".pkl")
        if os.path.exists(vecnorm_path) and vecnorm_kwargs:
            print(f"Loading VecNormalize stats from: {vecnorm_path}")
            from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

            if isinstance(venv, VecNormalize):
                print("Unwrapping existing VecNormalize wrapper to avoid nesting...")
                venv = venv.venv

            venv = VecNormalize.load(vecnorm_path, venv)
            venv.training = True
            venv.norm_reward = True
            if hasattr(venv, "obs_rms"):
                eval_env.obs_rms = venv.obs_rms
            if hasattr(venv, "ret_rms"):
                eval_env.ret_rms = venv.ret_rms
            print("VecNormalize statistics loaded successfully.")

        model = SAC.load(
            cfg.resume_from,
            env=venv,
            verbose=1,
        )
        model.set_logger(logger)

        # Calculate remaining timesteps
        current_timesteps = model.num_timesteps
        target_timesteps = int(cfg.training.total_timesteps)
        remaining_timesteps = max(0, target_timesteps - current_timesteps)

        print(f"Current timesteps: {current_timesteps:,}")
        print(f"Target timesteps: {target_timesteps:,}")
        print(f"Remaining timesteps: {remaining_timesteps:,}")
        print("Model loaded successfully. Continuing training...\n")
    else:
        # Create new SAC model
        remaining_timesteps = int(cfg.training.total_timesteps)

        # Build policy kwargs (convert OmegaConf to native Python dict)
        policy_kwargs = OmegaConf.to_container(cfg.algo.policy_kwargs, resolve=True) if cfg.algo.get("policy_kwargs") else {}

        model = SAC(
            policy=cfg.algo.policy,
            env=venv,
            learning_rate=hp.learning_rate,
            buffer_size=int(hp.buffer_size),
            learning_starts=int(hp.learning_starts),
            batch_size=int(hp.batch_size),
            tau=hp.tau,
            gamma=hp.gamma,
            ent_coef=hp.ent_coef,
            train_freq=int(hp.train_freq),
            gradient_steps=int(hp.gradient_steps),
            target_update_interval=int(hp.target_update_interval),
            use_sde=hp.get("use_sde", False),
            use_sde_at_warmup=hp.get("use_sde_at_warmup", False),
            policy_kwargs=policy_kwargs,
            device=cfg.algo.get("device", "auto"),
            verbose=1,
        )
        model.set_logger(logger)

    # Checkpoint callback (saves model + VecNormalize)
    ckpt_cb = CheckpointAndVecNormCallback(
        save_dir=os.path.join(run_dir, "checkpoints"),
        save_freq_steps=int(cfg.training.checkpoint_every_steps),
        verbose=1,
    )

    # Train the agent
    print("\n=== Starting SAC Training ===")
    if resuming:
        if remaining_timesteps <= 0:
            print(f"[INFO] Target of {int(cfg.training.total_timesteps):,} timesteps already reached. Skipping training.")
        else:
            print(f"Training for {remaining_timesteps:,} more timesteps to reach target.")
    start_time = time.time()

    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[ckpt_cb, eval_cb],
            log_interval=cfg.training.log_interval,
            reset_num_timesteps=False,
        )

    end_time = time.time()
    training_duration = end_time - start_time

    # Final saves
    model.save(os.path.join(run_dir, "final_model.zip"))
    try:
        venv.save(os.path.join(run_dir, "vecnormalize_final.pkl"))
    except Exception as e:
        print(f"[WARN] VecNormalize final save failed: {e}")

    # Print timing summary
    print("\n" + "="*60)
    print("SAC Training Complete!")
    print("="*60)
    print(f"Total training time: {timedelta(seconds=int(training_duration))}")
    print(f"Total timesteps: {int(cfg.training.total_timesteps):,}")
    print(f"Timesteps per second: {int(cfg.training.total_timesteps) / training_duration:.2f}")
    print(f"Run directory: {run_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Override Hydra's default output directory to outputs_sac
    sys.argv.append('hydra.run.dir=outputs_sac/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    
    main()
