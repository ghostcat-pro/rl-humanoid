# evaluate_torchrl.py
# Evaluate and visualize trained TorchRL PPO policies

import argparse
import sys
import os
import torch
import torch.nn as nn
import glob
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
    NormalParamExtractor,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)
from torchrl.envs import GymEnv
from torchrl.envs.transforms import TransformedEnv, DoubleToFloat

torch.set_default_dtype(torch.float32)


def mlp(in_dim: int, out_dim: int, hidden=(256, 256), act=nn.Tanh):
    """Create a multi-layer perceptron."""
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class CastObs(nn.Module):
    """Cast observation to float32."""
    def forward(self, x: torch.Tensor):
        return x.to(torch.float32)


def make_env(env_id: str, device: torch.device, render_mode="human"):
    """Create a GymEnv with float32 observations and rendering."""
    env = GymEnv(env_id, device=device, render_mode=render_mode)
    env = TransformedEnv(env)
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


def build_policy(obs_dim: int, act_dim: int, hidden=(256, 256), device="cpu"):
    """Build the policy network (same architecture as training)."""
    device = torch.device(device)
    
    # Cast module
    cast_module = TensorDictModule(
        module=CastObs(),
        in_keys=["observation"],
        out_keys=["observation"],
    )
    
    # Policy backbone: observation -> (loc, scale)
    policy_backbone = mlp(obs_dim, 2 * act_dim, hidden=hidden)
    dist_param_module = TensorDictModule(
        module=nn.Sequential(policy_backbone, NormalParamExtractor()),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    
    # Distribution module with Independent Normal
    from torch.distributions import Independent, Normal
    
    def make_independent_normal(loc, scale, **kwargs):
        return Independent(Normal(loc, scale), 1)
    
    prob_module = ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=make_independent_normal,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    
    # Full policy pipeline
    policy = ProbabilisticTensorDictSequential(cast_module, dist_param_module, prob_module)
    policy.to(device)
    
    return policy


def find_latest_checkpoint(base_dir: str, env_id: str, checkpoint_type: str = "final"):
    """Find the latest checkpoint for a given environment."""
    pattern = os.path.join(base_dir, env_id, "*", "*")
    
    if checkpoint_type == "final":
        pattern = os.path.join(base_dir, env_id, "*", "*", "checkpoint_final.pt")
    else:
        pattern = os.path.join(base_dir, env_id, "*", "*", "checkpoints", "*.pt")
    
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    # Sort by modification time, return most recent
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def evaluate_policy(policy, env, n_episodes: int = 5, deterministic: bool = True, render: bool = True):
    """Evaluate a policy for multiple episodes."""
    policy.eval()
    episode_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for ep in range(n_episodes):
            td = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            print(f"\n{'='*60}")
            print(f"Episode {ep + 1}/{n_episodes}")
            print(f"{'='*60}")
            
            while not done:
                # Get action from policy
                if deterministic:
                    # Use mean of distribution (loc)
                    with torch.no_grad():
                        td_out = policy[:-1](td)  # All modules except probabilistic sampling
                        action = td_out["loc"]
                        td.set("action", action)
                else:
                    # Sample from distribution
                    td = policy(td)
                
                # Step environment
                td = env.step(td)
                
                # Get reward and done status
                reward = td["next", "reward"].item()
                done = td["next", "done"].item()
                
                episode_reward += reward
                episode_length += 1
                
                # Move to next state (update td with next state)
                if not done:
                    # Copy next state to current state for next iteration
                    for key in td.keys():
                        if ("next", key) in td.keys(True):
                            td.set(key, td["next", key])
                
                if render:
                    try:
                        env.render()
                    except Exception as e:
                        print(f"Warning: Rendering failed: {e}")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length} steps")
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TorchRL PPO policy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--env_id", type=str, default="Walker2d-v5", help="Environment ID")
    parser.add_argument("--logdir", type=str, default="outputs_torchrl", help="Base log directory")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (mean)")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--hidden", type=str, default="256,256", help="Hidden layer sizes")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    hidden_sizes = tuple([int(x) for x in args.hidden.split(",") if x.strip()])
    
    # Find checkpoint if not specified
    if args.checkpoint is None:
        print(f"🔍 Searching for latest checkpoint in {args.logdir}/{args.env_id}...")
        args.checkpoint = find_latest_checkpoint(args.logdir, args.env_id)
        if args.checkpoint is None:
            print(f"❌ No checkpoint found for {args.env_id} in {args.logdir}")
            print(f"   Train a model first with: python torchrl_train_ppo.py --env_id {args.env_id}")
            return
        print(f"✅ Found: {args.checkpoint}")
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Print checkpoint info
    if "args" in checkpoint:
        print(f"   Trained for {checkpoint.get('frames_done', 'unknown')} frames")
        print(f"   Batch: {checkpoint.get('batch_num', 'unknown')}")
    
    # Create environment
    render_mode = None if args.no_render else "human"
    print(f"\n🌍 Creating environment: {args.env_id}")
    env = make_env(args.env_id, device=device, render_mode=render_mode)
    
    # Get observation and action dimensions
    obs_spec = env.observation_spec
    if hasattr(obs_spec, "keys") and "observation" in obs_spec.keys():
        obs_dim = obs_spec["observation"].shape[-1]
    else:
        obs_dim = obs_spec.shape[-1]
    act_dim = env.action_spec.shape[-1]
    
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dim: {act_dim}")
    
    # Build policy
    print(f"\n🤖 Building policy network...")
    policy = build_policy(obs_dim, act_dim, hidden=hidden_sizes, device=device)
    
    # Load policy weights
    policy.load_state_dict(checkpoint["policy_state_dict"])
    print(f"✅ Policy weights loaded successfully")
    
    # Evaluate
    print(f"\n🎬 Starting evaluation...")
    print(f"   Episodes: {args.n_episodes}")
    print(f"   Mode: {'Deterministic (mean)' if args.deterministic else 'Stochastic (sampling)'}")
    print(f"   Rendering: {'Enabled' if not args.no_render else 'Disabled'}")
    
    results = evaluate_policy(
        policy, 
        env, 
        n_episodes=args.n_episodes, 
        deterministic=args.deterministic,
        render=not args.no_render
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length:  {results['mean_length']:.1f} steps")
    print(f"\nEpisode Rewards: {[f'{r:.2f}' for r in results['episode_rewards']]}")
    print(f"Episode Lengths: {results['episode_lengths']}")
    print(f"{'='*60}\n")
    
    env.close()


if __name__ == "__main__":
    main()
