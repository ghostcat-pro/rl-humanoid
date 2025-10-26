from __future__ import annotations
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
    """
    Train a PPO agent on a Gymnasium MuJoCo environment using Hydra configs.
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

    # Configure logger (Hydra gives each run a unique output dir)
    run_dir = cfg.paths.log_root
    logger = configure(run_dir, ["stdout", "tensorboard", "csv"])

    # Validate batch size divisibility
    hp = cfg.algo.hyperparams
    n_steps = int(hp.n_steps)
    n_envs = int(cfg.env.vec_env.n_envs)
    batch_size = int(hp.batch_size)
    total_samples = n_steps * n_envs
    assert total_samples % batch_size == 0, (
        f"Invalid batch_size={batch_size}: must divide n_envs*n_steps={total_samples}"
    )

    # Instantiate PPO
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

    # Checkpoint callback (saves model + VecNormalize)
    ckpt_cb = CheckpointAndVecNormCallback(
        save_dir=os.path.join(run_dir, "checkpoints"),
        save_freq_steps=int(cfg.training.checkpoint_every_steps),
        verbose=1,
    )

    # Train the agent
    model.learn(
        total_timesteps=int(cfg.training.total_timesteps),
        callback=ckpt_cb,
        log_interval=cfg.training.log_interval,
    )

    # Final saves
    model.save(os.path.join(run_dir, "final_model.zip"))
    try:
        venv.save(os.path.join(run_dir, "vecnormalize_final.pkl"))
    except Exception as e:
        print(f"[WARN] VecNormalize final save failed: {e}")

    print(f"\nRun directory: {run_dir}\n✅ Training complete.\n")


if __name__ == "__main__":
    main()
