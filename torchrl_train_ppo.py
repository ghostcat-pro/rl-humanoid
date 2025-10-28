# torchrl_train_ppo.py
# Minimal PPO for continuous Gymnasium envs (Walker2d-v5 / Humanoid-v5) using TorchRL 0.5+
# - Robust to TorchRL API variations (no ProbabilisticActor dependency)
# - Fixes dtype mismatch by casting obs to float32 in both actor and critic paths
# - Adds gradient clipping

import argparse
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
    NormalParamExtractor,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)

from torchrl.envs import GymEnv, ParallelEnv
from torchrl.envs.transforms import TransformedEnv, DoubleToFloat
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ValueOperator
from torchrl.objectives import ClipPPOLoss

# Ensure default dtype is float32 (helps avoid silent float64 tensors)
torch.set_default_dtype(torch.float32)


def mlp(in_dim: int, out_dim: int, hidden=(256, 256), act=nn.Tanh):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class CastObs(nn.Module):
    """Cast observation to float32 to avoid dtype mismatch with model weights."""
    def forward(self, x: torch.Tensor):
        return x.to(torch.float32)


class CastToFloat32Transform:
    """TorchRL transform that casts observation tensors to float32."""
    def __init__(self):
        pass
    
    def __call__(self, tensordict):
        if "observation" in tensordict.keys():
            tensordict.set("observation", tensordict.get("observation").to(torch.float32))
        return tensordict


def make_env(env_id: str, device: torch.device):
    """Create a GymEnv with float32 observations."""
    env = GymEnv(env_id, device=device)
    # Wrap with transform to cast observations from float64 to float32
    env = TransformedEnv(env)
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


class SampleNormalModule(nn.Module):
    """Takes (loc, scale) and returns (action, sample_log_prob)."""
    def forward(self, loc: torch.Tensor, scale: torch.Tensor):
        dist = torch.distributions.Normal(loc, scale)
        action = dist.rsample()
        logp = dist.log_prob(action).sum(-1, keepdim=True)  # [N,1]
        return action, logp


class SumLogProb(nn.Module):
    """Sum log_prob across action dimensions."""
    def forward(self, sample_log_prob: torch.Tensor):
        if sample_log_prob.dim() > 1 and sample_log_prob.shape[-1] > 1:
            return sample_log_prob.sum(-1, keepdim=True)
        return sample_log_prob


def compute_gae(rewards, dones, values, next_values, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation on a flat [N] trajectory."""
    N = rewards.size(0)
    adv = torch.zeros_like(rewards)
    gae = 0.0
    dones_f = dones.to(torch.float32)
    for t in reversed(range(N)):
        nonterminal = 1.0 - dones_f[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="Humanoid-v5")
    p.add_argument("--total_frames", type=int, default=5_000_000)
    p.add_argument("--frames_per_batch", type=int, default=8192)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)  # used inside ClipPPOLoss
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--logdir", type=str, default=None, help="Base directory for logs (default: outputs_torchrl)")
    p.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N batches")
    args = p.parse_args()

    device = torch.device(args.device)

    # ----- Create timestamped output directory -----
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    if args.logdir:
        base_dir = args.logdir
    else:
        base_dir = "outputs_torchrl"
    
    log_dir = os.path.join(base_dir, args.env_id, timestamp)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"📁 Logging to: {log_dir}")
    
    # Save training configuration
    config_path = os.path.join(log_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("Training Configuration\n")
        f.write("=" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Initialize TensorBoard writer
    tb_writer = SummaryWriter(log_dir=log_dir)

    # ----- Probe specs (handles composite observation) -----
    probe = make_env(args.env_id, device=device)
    obs_spec = probe.observation_spec
    if hasattr(obs_spec, "keys") and "observation" in obs_spec.keys():
        obs_dim = obs_spec["observation"].shape[-1]
    else:
        obs_dim = obs_spec.shape[-1]
    act_dim = probe.action_spec.shape[-1]
    print(f"\n🌍 Env: {args.env_id} | obs_dim={obs_dim}, act_dim={act_dim}, device={device}")

    # Warm-up parallel env ctor (optional)
    _ = ParallelEnv(args.n_envs, lambda: make_env(args.env_id, device=device))

    # -------- Shared cast module (obs -> float32) --------
    cast_module = TensorDictModule(
        module=CastObs(),
        in_keys=["observation"],
        out_keys=["observation"],
    )

    # -------- Policy --------
    # observation -> (loc, scale)
    policy_backbone = mlp(obs_dim, 2 * act_dim)
    dist_param_module = TensorDictModule(
        module=nn.Sequential(policy_backbone, NormalParamExtractor()),  # ensures positive scale
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    # (loc, scale) -> distribution -> action
    from torch.distributions import Independent, Normal
    
    # Use a lambda to wrap Normal in Independent for summed log probs
    def make_independent_normal(loc, scale, **kwargs):
        return Independent(Normal(loc, scale), 1)
    
    prob_module_indep = ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=make_independent_normal,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )

    # Full policy pipeline: cast -> dist params -> sample
    policy = ProbabilisticTensorDictSequential(cast_module, dist_param_module, prob_module_indep)

    # -------- Critic --------
    value = ValueOperator(module=mlp(obs_dim, 1), in_keys=["observation"])
    # Compose a value pipeline with the same cast in front
    value_seq = TensorDictSequential(cast_module, value)

    # -------- Collector --------
    collector = SyncDataCollector(
        create_env_fn=lambda: make_env(args.env_id, device=device),
        policy=policy,  # must write "action"
        total_frames=args.total_frames,
        frames_per_batch=args.frames_per_batch,
        device=device,
    )

    # -------- PPO objective --------
    loss_mod = ClipPPOLoss(
        actor=policy,          # produces action + sample_log_prob
        critic=value_seq,      # produces state_value
        clip_epsilon=0.2,
        entropy_coef=args.ent_coef,      # included in losses["loss_entropy"]
        normalize_advantage=True,
    )
    optim = torch.optim.Adam(loss_mod.parameters(), lr=args.lr)

    progress = tqdm(total=args.total_frames, desc=f"Training {args.env_id}")
    frames_done = 0
    batch_num = 0

    for batch in collector:
        batch_num += 1
        # batch has batch_size [T, B] or [N]; flatten to [N] for SGD
        batch = batch.to(device)
        if len(batch.batch_size) == 2:
            T, B = batch.batch_size
            flat = batch.reshape(T * B)
        else:
            flat = batch

        # Core tensors - use ("next", "reward") for the reward key
        rewards = flat.get(("next", "reward")).squeeze(-1)                # [N]
        dones = flat.get(("next", "done")).squeeze(-1).to(torch.bool)     # [N]
        next_obs = flat["next", "observation"]                  # [N, obs_dim]

        with torch.no_grad():
            values = value_seq(flat)["state_value"].squeeze(-1)     # [N]
            nv = value_seq(TensorDict({"observation": next_obs}, batch_size=[next_obs.shape[0]]))["state_value"]
            next_values = nv.squeeze(-1)                              # [N]

        adv, rets = compute_gae(rewards, dones, values, next_values)
        flat.set("advantage", adv.unsqueeze(-1))  # [N,1] for PPO
        flat.set("value_target", rets.unsqueeze(-1))  # [N,1] for PPO

        # Ensure sample_log_prob is [N,1] (Independent Normal gives [N])
        if "sample_log_prob" in flat.keys() and flat["sample_log_prob"].dim() == 1:
            flat.set("sample_log_prob", flat["sample_log_prob"].unsqueeze(-1))

        # ----- PPO epochs with minibatching -----
        N = flat.batch_size[0]
        for _ in range(args.epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(N)), args.minibatch, drop_last=True)
            for idx in sampler:
                mb = flat[idx]

                # Ensure sample_log_prob is [N,1]
                if "sample_log_prob" in mb.keys() and mb["sample_log_prob"].dim() == 1:
                    mb.set("sample_log_prob", mb["sample_log_prob"].unsqueeze(-1))

                losses = loss_mod(mb)
                loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]

                optim.zero_grad()
                loss.backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), args.max_grad_norm)
                optim.step()

        frames_done += args.frames_per_batch
        progress.update(args.frames_per_batch)
        
        # Calculate metrics
        avg_reward = float(rewards.mean())
        progress.set_postfix({"avg_reward": avg_reward})
        
        # Log to TensorBoard
        tb_writer.add_scalar("time/total_frames", frames_done, batch_num)
        tb_writer.add_scalar("rollout/reward_mean_step", avg_reward, batch_num)
        tb_writer.add_scalar("loss/objective", losses["loss_objective"].item(), batch_num)
        tb_writer.add_scalar("loss/critic", losses["loss_critic"].item(), batch_num)
        tb_writer.add_scalar("loss/entropy", losses["loss_entropy"].item(), batch_num)
        tb_writer.add_scalar("loss/total", loss.item(), batch_num)
        
        # Save checkpoint periodically
        if batch_num % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_batch_{batch_num}.pt")
            torch.save({
                'batch_num': batch_num,
                'frames_done': frames_done,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_seq.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")

        if frames_done >= args.total_frames:
            break

    progress.close()
    tb_writer.close()
    
    print("\n✅ Training complete.")
    
    # Save final models
    final_policy_path = os.path.join(log_dir, "policy_final.pt")
    final_value_path = os.path.join(log_dir, "value_final.pt")
    final_checkpoint_path = os.path.join(log_dir, "checkpoint_final.pt")
    
    torch.save(policy.state_dict(), final_policy_path)
    torch.save(value_seq.state_dict(), final_value_path)
    torch.save({
        'batch_num': batch_num,
        'frames_done': frames_done,
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value_seq.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'args': vars(args),
    }, final_checkpoint_path)
    
    print(f"📦 Final models saved to: {log_dir}")
    print(f"   - Policy: {final_policy_path}")
    print(f"   - Value: {final_value_path}")
    print(f"   - Checkpoint: {final_checkpoint_path}")
    print(f"\n📊 View training logs with: tensorboard --logdir {base_dir}")


if __name__ == "__main__":
    main()
