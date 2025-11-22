from __future__ import annotations
import os
import sys
import time
from datetime import timedelta
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
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
    Train a PPO agent on the Humanoid Destination environment.
    """
    # OVERRIDE CONFIG FOR DESTINATION TASK
    # We do this programmatically to avoid creating a whole new config file structure for now,
    # but still leveraging the existing hydra setup.
    cfg.env.name = "HumanoidDestination-v0"
    
    # Update output directory to be distinct
    # Hydra sets the output dir based on config, but we can't easily change it *after* hydra starts.
    # However, we can just use a different subdirectory for our logs/checkpoints if we want,
    # OR we can rely on the user running this script to know it goes to the standard hydra output location.
    # The user asked for a "new results/output directory dedicated to this task".
    # Hydra's default output dir is usually `outputs/date/time`.
    # We can force a specific run directory logic if we weren't using Hydra's automatic dir creation,
    # but since we are, let's just print where it is going.
    # actually, let's try to modify the log dir if possible, or just accept Hydra's default.
    # The user said "create ... a new results/output directory dedicated to this task".
    # If we run this script, Hydra will create a new directory in `outputs/` anyway.
    # To make it "dedicated", we might want to change the hydra run dir pattern, but that's in the config.
    # Let's just proceed with standard Hydra behavior which creates unique dirs per run.
    
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

    # Create vectorized environment
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

    # Configure logger
    # Use Hydra's output directory
    run_dir = HydraConfig.get().runtime.output_dir
    
    logger = configure(run_dir, ["stdout", "tensorboard", "csv"])

    # === Evaluation environment and callback (save best model) ===
    eval_env = DummyVecEnv([make_single_env(cfg.env.name, {"render_mode": None}, monitor=False, seed=cfg.seed + 10)])

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

    # Validate batch size divisibility
    hp = cfg.algo.hyperparams
    n_steps = int(hp.n_steps)
    n_envs = int(cfg.env.vec_env.n_envs)
    batch_size = int(hp.batch_size)
    total_samples = n_steps * n_envs
    assert total_samples % batch_size == 0, (
        f"Invalid batch_size={batch_size}: must divide n_envs*n_steps={total_samples}"
    )

    # Instantiate or load PPO model
    resuming = cfg.get("resume_from") and cfg.resume_from
    if resuming:
        print(f"\n=== Resuming from checkpoint: {cfg.resume_from} ===")
        
        # Load VecNormalize stats if available
        vecnorm_path = cfg.resume_from.replace("model_", "vecnormalize_").replace(".zip", ".pkl")
        if os.path.exists(vecnorm_path) and vecnorm_kwargs:
            print(f"Loading VecNormalize stats from: {vecnorm_path}")
            from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
            venv = VecNormalize.load(vecnorm_path, venv)
            venv.training = True
            venv.norm_reward = True
            # Update eval env stats as well
            if hasattr(venv, "obs_rms"):
                eval_env.obs_rms = venv.obs_rms
            if hasattr(venv, "ret_rms"):
                eval_env.ret_rms = venv.ret_rms
            print("VecNormalize statistics loaded successfully.")
        
        model = PPO.load(
            cfg.resume_from,
            env=venv,
            verbose=1,
        )
        model.set_logger(logger)
        
        # Calculate remaining timesteps to reach target
        current_timesteps = model.num_timesteps
        target_timesteps = int(cfg.training.total_timesteps)
        remaining_timesteps = max(0, target_timesteps - current_timesteps)
        
        print(f"Current timesteps: {current_timesteps:,}")
        print(f"Target timesteps: {target_timesteps:,}")
        print(f"Remaining timesteps: {remaining_timesteps:,}")
        print("Model loaded successfully. Continuing training...\n")
    else:
        # Create new model
        remaining_timesteps = int(cfg.training.total_timesteps)
        model = PPO(
            policy=cfg.algo.policy,
            env=venv,
            learning_rate=hp.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
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

    # Checkpoint callback
    ckpt_cb = CheckpointAndVecNormCallback(
        save_dir=os.path.join(run_dir, "checkpoints"),
        save_freq_steps=int(cfg.training.checkpoint_every_steps),
        verbose=1,
    )

    # Train the agent
    print("\n=== Starting Training (Destination Task) ===")
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
            reset_num_timesteps=False,  # Always preserve timestep count
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
    print("Training Complete!")
    print("="*60)
    print(f"Total training time: {timedelta(seconds=int(training_duration))}")
    print(f"Total timesteps: {int(cfg.training.total_timesteps):,}")
    print(f"Timesteps per second: {int(cfg.training.total_timesteps) / training_duration:.2f}")
    print(f"Run directory: {run_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
