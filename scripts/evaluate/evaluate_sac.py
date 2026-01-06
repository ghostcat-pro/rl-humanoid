"""
Evaluate and visualize a trained SAC model.

Usage:
    python scripts/evaluate/evaluate_sac.py --model_path <path> --render
    python scripts/evaluate/evaluate_sac.py --model_path <path> --save_video
"""
from __future__ import annotations
import argparse
import os
import sys
import json
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.make_env import make_single_env
from utils.vecnorm_io import maybe_load_vecnormalize

# Import custom environments to register them
import envs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="Humanoid-v5")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--vecnorm_path", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true", help="Render in window")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--env_kwargs", type=str, default=None,
                    help="JSON string of environment kwargs")
    ap.add_argument("--save_video", action="store_true", help="Save video of episodes")
    ap.add_argument("--video_folder", type=str, default="videos", help="Folder to save videos")
    ap.add_argument("--video_length", type=int, default=0, help="Length of recorded video (0 = full episode)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Rendering mode
    if args.save_video:
        make_kwargs = {"render_mode": "rgb_array"}
    elif args.render:
        make_kwargs = {"render_mode": "human"}
    else:
        make_kwargs = {"render_mode": None}

    # Add any additional env_kwargs from command line
    if args.env_kwargs:
        try:
            extra_kwargs = json.loads(args.env_kwargs)
            make_kwargs.update(extra_kwargs)
            print(f"Using environment kwargs: {make_kwargs}")
        except json.JSONDecodeError as e:
            print(f"Error parsing env_kwargs: {e}")
            sys.exit(1)

    # Single-env VecEnv for evaluation
    env_fn = make_single_env(args.env_id, make_kwargs, monitor=False, seed=args.seed)
    venv = DummyVecEnv([env_fn])

    # Optionally load VecNormalize stats
    if args.vecnorm_path:
        venv = maybe_load_vecnormalize(venv, args.vecnorm_path)

    # Wrap with video recorder if requested
    if args.save_video:
        os.makedirs(args.video_folder, exist_ok=True)
        venv = VecVideoRecorder(
            venv,
            args.video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=args.video_length if args.video_length > 0 else 100000,
            name_prefix="sac_eval"
        )
        print(f"Recording video to {args.video_folder}")

    # Load the trained SAC model
    model = SAC.load(args.model_path)

    print(f"\nEvaluating SAC model: {args.model_path}")
    print(f"Environment: {args.env_id}")
    print(f"Episodes: {args.episodes}")
    print("-" * 40)

    try:
        total_reward = 0
        for ep in range(args.episodes):
            obs = venv.reset()
            done = False
            ep_rew = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, rewards, dones, infos = venv.step(action)
                ep_rew += float(rewards[0])
                steps += 1
                done = bool(dones[0])
            total_reward += ep_rew
            print(f"Episode {ep+1}: reward={ep_rew:.2f}, steps={steps}")

        print("-" * 40)
        print(f"Average reward: {total_reward / args.episodes:.2f}")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
