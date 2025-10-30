from __future__ import annotations
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.make_env import make_single_env
from utils.vecnorm_io import maybe_load_vecnormalize


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="Humanoid-v5")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--vecnorm_path", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--verbose", action="store_true", help="Print step-by-step info")
    return ap.parse_args()


def main():
    args = parse_args()

    # No rendering - pure statistics
    make_kwargs = {"render_mode": None}

    # Single-env VecEnv for evaluation
    env_fn = make_single_env(args.env_id, make_kwargs, monitor=False, seed=args.seed)
    venv = DummyVecEnv([env_fn])

    # Optionally load VecNormalize stats (eval mode, no reward norm)
    if args.vecnorm_path:
        venv = maybe_load_vecnormalize(venv, args.vecnorm_path)

    # Load the trained policy
    model = PPO.load(args.model_path)

    print(f"\n{'='*60}")
    print(f"Evaluating: {args.env_id}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"{'='*60}\n")

    try:
        rewards = []
        episode_lengths = []
        
        for ep in range(args.episodes):
            obs = venv.reset()
            done = False
            ep_rew = 0.0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, rewards_step, dones, infos = venv.step(action)
                ep_rew += float(rewards_step[0])
                done = bool(dones[0])
                steps += 1
                
                if args.verbose and steps % 100 == 0:
                    print(f"  Episode {ep+1}, Step {steps}: reward so far = {ep_rew:.2f}")
            
            rewards.append(ep_rew)
            episode_lengths.append(steps)
            print(f"Episode {ep+1:2d}: reward={ep_rew:8.2f}, steps={steps:4d}")
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Episodes:        {len(rewards)}")
        print(f"Mean reward:     {np.mean(rewards):8.2f} ± {np.std(rewards):.2f}")
        print(f"Min reward:      {np.min(rewards):8.2f}")
        print(f"Max reward:      {np.max(rewards):8.2f}")
        print(f"Median reward:   {np.median(rewards):8.2f}")
        print(f"Mean length:     {np.mean(episode_lengths):8.2f} steps")
        print(f"{'='*60}\n")
        
        # Additional statistics
        print(f"Reward distribution:")
        print(f"  25th percentile: {np.percentile(rewards, 25):8.2f}")
        print(f"  50th percentile: {np.percentile(rewards, 50):8.2f}")
        print(f"  75th percentile: {np.percentile(rewards, 75):8.2f}")
        print(f"\nSuccess rate (reward > 5000): {100 * sum(r > 5000 for r in rewards) / len(rewards):.1f}%")
        
    finally:
        venv.close()


if __name__ == "__main__":
    main()
