from __future__ import annotations
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def main():
    args = parse_args()

    # Rendering is controlled via the env's make_kwargs
    make_kwargs = {"render_mode": "human"} if args.render else {"render_mode": None}

    # Single-env VecEnv for evaluation
    env_fn = make_single_env(args.env_id, make_kwargs, monitor=False, seed=args.seed)
    venv = DummyVecEnv([env_fn])

    # Optionally load VecNormalize stats (eval mode, no reward norm)
    if args.vecnorm_path:
        venv = maybe_load_vecnormalize(venv, args.vecnorm_path)

    # Load the trained policy
    model = PPO.load(args.model_path)

    try:
        for ep in range(args.episodes):
            obs = venv.reset()
            done = False
            ep_rew = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, rewards, dones, infos = venv.step(action)
                ep_rew += float(rewards[0])
                done = bool(dones[0])
            print(f"Episode {ep+1}: reward={ep_rew:.2f}")
    finally:
        venv.close()   # <— ensures viewer and GLFW clean up


if __name__ == "__main__":
    main()
