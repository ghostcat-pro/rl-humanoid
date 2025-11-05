"""Evaluate trained HumanoidStairs agent."""

import argparse
import os
import glob
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.make_env import make_single_env
from utils.vecnorm_io import maybe_load_vecnormalize
import envs  # Register custom environments


def find_latest_run():
    """Find the most recent run directory."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None

    # Find all run directories with dates
    run_dirs = []
    for date_dir in outputs_dir.iterdir():
        if date_dir.is_dir():
            for time_dir in date_dir.iterdir():
                if time_dir.is_dir():
                    run_dirs.append(time_dir)

    if not run_dirs:
        return None

    # Sort by modification time, most recent first
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def find_best_model(run_dir, use_final=False):
    """Find the best or final model in the run directory."""
    run_path = Path(run_dir)

    if use_final:
        model_path = run_path / "final_model.zip"
        vecnorm_path = run_path / "vecnormalize_final.pkl"
    else:
        # Look for best model in eval directory
        eval_dir = run_path / "eval"
        if eval_dir.exists():
            best_model = eval_dir / "best_model.zip"
            if best_model.exists():
                print(f"✅ Found best model: {best_model}")
                # Try to find matching vecnorm
                vecnorm_candidates = list(run_path.glob("vecnormalize_*.pkl"))
                if vecnorm_candidates:
                    vecnorm_path = max(vecnorm_candidates, key=lambda x: x.stat().st_mtime)
                else:
                    vecnorm_path = run_path / "vecnormalize_final.pkl"
                return best_model, vecnorm_path if vecnorm_path.exists() else None

        # Fallback to latest checkpoint
        checkpoints = list(run_path.glob("checkpoints/model_*.zip"))
        if checkpoints:
            model_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
            # Try to match vecnorm
            model_steps = model_path.stem.split("_")[1]
            vecnorm_path = run_path / f"checkpoints/vecnormalize_{model_steps}.pkl"
            if not vecnorm_path.exists():
                vecnorm_path = run_path / "vecnormalize_final.pkl"
        else:
            model_path = run_path / "final_model.zip"
            vecnorm_path = run_path / "vecnormalize_final.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return model_path, vecnorm_path if vecnorm_path.exists() else None


def evaluate(model_path, vecnorm_path, env_id, episodes=5, render=True, deterministic=True, seed=123):
    """Evaluate a trained model."""
    print(f"\n{'='*60}")
    print(f"Evaluating HumanoidStairs Agent")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    if vecnorm_path:
        print(f"VecNorm: {vecnorm_path}")
    else:
        print(f"⚠️  No VecNormalize file found (may affect performance)")
    print(f"Environment: {env_id}")
    print(f"Episodes: {episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*60}\n")

    # Create environment
    make_kwargs = {"render_mode": "human"} if render else {"render_mode": None}
    env_fn = make_single_env(env_id, make_kwargs, monitor=False, seed=seed)
    venv = DummyVecEnv([env_fn])

    # Load VecNormalize if available
    if vecnorm_path:
        venv = maybe_load_vecnormalize(venv, vecnorm_path)

    # Load model
    model = PPO.load(model_path)

    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    max_heights = []
    max_distances = []

    try:
        for ep in range(episodes):
            obs = venv.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            max_height = 0.0
            max_distance = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = venv.step(action)
                ep_reward += float(rewards[0])
                ep_length += 1
                done = bool(dones[0])

                # Track stats
                if infos and len(infos) > 0:
                    info = infos[0]
                    if 'z_position' in info:
                        max_height = max(max_height, info['z_position'])
                    if 'x_position' in info:
                        max_distance = max(max_distance, info['x_position'])

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            max_heights.append(max_height)
            max_distances.append(max_distance)

            print(f"Episode {ep+1:2d}: reward={ep_reward:7.2f} | length={ep_length:4d} | "
                  f"max_height={max_height:.2f}m | max_distance={max_distance:.2f}m")

        print(f"\n{'='*60}")
        print(f"📊 Evaluation Results ({episodes} episodes)")
        print(f"{'='*60}")
        print(f"Mean reward:        {sum(episode_rewards)/len(episode_rewards):7.2f} ± {max(episode_rewards) - min(episode_rewards):7.2f}")
        print(f"Mean episode length: {sum(episode_lengths)/len(episode_lengths):6.1f}")
        print(f"Mean max height:     {sum(max_heights)/len(max_heights):6.2f}m")
        print(f"Mean max distance:   {sum(max_distances)/len(max_distances):6.2f}m")
        print(f"{'='*60}\n")

    finally:
        venv.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate HumanoidStairs model")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory path")
    parser.add_argument("--model-path", type=str, default=None, help="Direct path to model")
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Direct path to vecnorm")
    parser.add_argument("--env-id", type=str, default="HumanoidStairs-v0", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--no-deterministic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--use-final", action="store_true", help="Use final model instead of best")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    # Determine model and vecnorm paths
    if args.model_path:
        model_path = Path(args.model_path)
        vecnorm_path = Path(args.vecnorm_path) if args.vecnorm_path else None
    else:
        # Find run directory
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_dir = find_latest_run()
            if run_dir is None:
                print("❌ No run directory found in ./outputs/")
                return
            print(f"📁 Using latest run: {run_dir}")

        # Find model and vecnorm
        model_path, vecnorm_path = find_best_model(run_dir, use_final=args.use_final)

    # Run evaluation
    evaluate(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        env_id=args.env_id,
        episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.no_deterministic,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
