#!/usr/bin/env python3
"""
Evaluate a trained model with video recording capability.
"""
from __future__ import annotations
import argparse
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.make_env import make_single_env
from utils.vecnorm_io import maybe_load_vecnormalize

# Import custom environments
import envs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="HumanoidStairs-v0")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--vecnorm_path", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--video_dir", type=str, default="./videos")
    ap.add_argument("--video_length", type=int, default=1000, help="Steps per video file")
    return ap.parse_args()


def main():
    args = parse_args()

    # Create environment with rgb_array rendering for video
    make_kwargs = {"render_mode": "rgb_array"}
    env_fn = make_single_env(args.env_id, make_kwargs, monitor=False, seed=args.seed)
    venv = DummyVecEnv([env_fn])

    # Load VecNormalize stats if provided
    if args.vecnorm_path:
        venv = maybe_load_vecnormalize(venv, args.vecnorm_path)

    # Wrap with video recorder
    os.makedirs(args.video_dir, exist_ok=True)
    venv = VecVideoRecorder(
        venv,
        args.video_dir,
        record_video_trigger=lambda x: x == 0,  # Record from start
        video_length=args.video_length,
        name_prefix=f"eval-{args.env_id}"
    )

    # Load the trained policy
    model = PPO.load(args.model_path)

    print(f"\n{'='*60}")
    print(f"Evaluating: {args.model_path}")
    print(f"Environment: {args.env_id}")
    print(f"Episodes: {args.episodes}")
    print(f"Video directory: {args.video_dir}")
    print(f"{'='*60}\n")

    try:
        total_reward = 0.0
        for ep in range(args.episodes):
            obs = venv.reset()
            done = False
            ep_rew = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, rewards, dones, infos = venv.step(action)
                ep_rew += float(rewards[0])
                done = bool(dones[0])
                steps += 1

            total_reward += ep_rew
            print(f"Episode {ep+1}: reward={ep_rew:.2f}, steps={steps}")

        avg_reward = total_reward / args.episodes
        print(f"\n{'='*60}")
        print(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")
        print(f"Videos saved to: {args.video_dir}")
        print(f"{'='*60}\n")

    finally:
        venv.close()


if __name__ == "__main__":
    main()
