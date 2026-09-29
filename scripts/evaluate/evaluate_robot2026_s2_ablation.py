from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import envs  # noqa: F401  # Registers custom environments.
from utils.make_env import make_single_env
from utils.vecnorm_io import maybe_load_vecnormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ROBOT2026 camera-ready S2 ablation runs and write after-action reports."
    )
    parser.add_argument("--manifest", required=True, help="CSV manifest of runs to evaluate.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", default="reports/robot2026_camera_ready/s2_ablation")
    parser.add_argument("--strict", action="store_true", help="Fail if any manifest artifact is missing.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_config(config_path: Path) -> Any:
    cfg = OmegaConf.load(config_path)
    if "env" not in cfg or "make_kwargs" not in cfg.env:
        raise ValueError(f"Config does not contain env.make_kwargs: {config_path}")
    return cfg


def as_plain_container(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True)


def evaluate_run(row: dict[str, str], episodes: int, seed: int, deterministic: bool) -> list[dict[str, Any]]:
    model_path = Path(row["model_path"])
    vecnorm_path = Path(row["vecnorm_path"])
    config_path = Path(row["config_path"])
    cfg = load_config(config_path)

    make_kwargs = dict(as_plain_container(cfg.env.make_kwargs) or {})
    make_kwargs["render_mode"] = None
    num_steps = int(make_kwargs.get("num_steps", 0))

    env_fn = make_single_env(str(cfg.env.name), make_kwargs, monitor=False, seed=seed)
    venv = DummyVecEnv([env_fn])
    venv = maybe_load_vecnormalize(venv, str(vecnorm_path))
    model = PPO.load(str(model_path))

    episodes_out: list[dict[str, Any]] = []
    try:
        for episode_idx in range(episodes):
            obs = venv.reset()
            done = False
            total_reward = 0.0
            length = 0
            last_info: dict[str, Any] = {}
            max_step_reached = 0
            max_z = -np.inf

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, infos = venv.step(action)
                info = dict(infos[0])
                total_reward += float(reward[0])
                length += 1
                done = bool(dones[0])
                last_info = info
                max_step_reached = max(max_step_reached, int(info.get("max_step_reached", 0)))
                max_z = max(max_z, float(info.get("z_position", -np.inf)))

            success = bool(num_steps and max_step_reached >= num_steps)
            episodes_out.append(
                {
                    "arm": row["arm"],
                    "seed": int(row["seed"]),
                    "episode": episode_idx + 1,
                    "eval_seed": seed,
                    "total_reward": total_reward,
                    "episode_length": length,
                    "success": success,
                    "max_step_reached": max_step_reached,
                    "target_steps": num_steps,
                    "final_x": float(last_info.get("x_position", np.nan)),
                    "final_y": float(last_info.get("y_position", np.nan)),
                    "final_z": float(last_info.get("z_position", np.nan)),
                    "max_z": max_z,
                    "terminated_unhealthy": bool(last_info.get("TimeLimit.truncated", False)) is False,
                    "run_dir": row["run_dir"],
                    "model_path": row["model_path"],
                }
            )
    finally:
        venv.close()

    return episodes_out


def summarize(per_episode: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_episode:
        by_run[(str(row["arm"]), int(row["seed"]))].append(row)

    run_summaries: list[dict[str, Any]] = []
    for (arm, seed), rows in sorted(by_run.items()):
        rewards = [float(r["total_reward"]) for r in rows]
        lengths = [float(r["episode_length"]) for r in rows]
        successes = [bool(r["success"]) for r in rows]
        max_steps = [int(r["max_step_reached"]) for r in rows]
        final_x = [float(r["final_x"]) for r in rows]
        run_summaries.append(
            {
                "arm": arm,
                "seed": seed,
                "episodes": len(rows),
                "success_rate": 100.0 * sum(successes) / len(successes),
                "mean_reward": statistics.fmean(rewards),
                "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
                "mean_length": statistics.fmean(lengths),
                "mean_max_step_reached": statistics.fmean(max_steps),
                "mean_final_x": statistics.fmean(final_x),
                "run_dir": rows[0]["run_dir"],
            }
        )
    return run_summaries


def arm_summary(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_summaries:
        by_arm[str(row["arm"])].append(row)

    arms: list[dict[str, Any]] = []
    for arm, rows in sorted(by_arm.items()):
        success_rates = [float(r["success_rate"]) for r in rows]
        mean_rewards = [float(r["mean_reward"]) for r in rows]
        mean_lengths = [float(r["mean_length"]) for r in rows]
        arms.append(
            {
                "arm": arm,
                "seeds_completed": len(rows),
                "seeds": " ".join(str(r["seed"]) for r in sorted(rows, key=lambda x: int(x["seed"]))),
                "mean_success_rate": statistics.fmean(success_rates),
                "sd_success_rate": statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
                "mean_reward_across_seeds": statistics.fmean(mean_rewards),
                "sd_reward_across_seeds": statistics.stdev(mean_rewards) if len(mean_rewards) > 1 else 0.0,
                "mean_length_across_seeds": statistics.fmean(mean_lengths),
                "sd_length_across_seeds": statistics.stdev(mean_lengths) if len(mean_lengths) > 1 else 0.0,
            }
        )
    return arms


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.2f}"
            values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out) + "\n"


def write_report(
    path: Path,
    manifest_path: Path,
    completed: list[dict[str, str]],
    missing: list[dict[str, str]],
    run_summaries: list[dict[str, Any]],
    arms: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "episodes": args.episodes,
        "eval_seed": args.seed,
        "deterministic": args.deterministic,
        "completed_runs": len(completed),
        "missing_runs": len(missing),
    }
    lines = [
        "# S2 Ablation After-Action Report",
        "",
        "## Evaluation Protocol",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "Success criterion: `max_step_reached >= num_steps` from each run config.",
        "",
        "## Arm Summary",
        "",
        markdown_table(
            arms,
            [
                "arm",
                "seeds_completed",
                "seeds",
                "mean_success_rate",
                "sd_success_rate",
                "mean_reward_across_seeds",
                "sd_reward_across_seeds",
                "mean_length_across_seeds",
            ],
        ),
        "## Seed Summary",
        "",
        markdown_table(
            run_summaries,
            [
                "arm",
                "seed",
                "episodes",
                "success_rate",
                "mean_reward",
                "std_reward",
                "mean_length",
                "mean_max_step_reached",
                "mean_final_x",
            ],
        ),
        "## Missing Runs",
        "",
    ]
    if missing:
        lines.append(markdown_table(missing, ["arm", "seed", "run_dir", "notes"]))
    else:
        lines.append("_None._\n")
    lines.extend(
        [
            "## Conclusions",
            "",
            "- Fill in after inspecting the completed arm summary.",
            "- State whether world-frame termination prevented S2 learning under matched training.",
            "- State whether removing stair-specific shaping reduced S2 completion under matched training.",
            "",
            "## Paper Changes Supported",
            "",
            "- Replace single-seed S2 values with seed-level and across-seed summaries.",
            "- Clarify checkpoint selection and held-out re-evaluation protocol.",
            "- Moderate broad reward-shaping and S4 termination claims if results do not support them.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_missing(row: dict[str, str]) -> bool:
    required = ["model_path", "vecnorm_path", "config_path"]
    return any(not Path(row[field]).is_file() for field in required)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(manifest_path)
    completed = [row for row in manifest if not artifact_missing(row)]
    missing = [row for row in manifest if artifact_missing(row)]

    if args.strict and missing:
        missing_labels = ", ".join(f"{row['arm']} seed {row['seed']}" for row in missing)
        raise SystemExit(f"Missing required artifacts: {missing_labels}")

    per_episode: list[dict[str, Any]] = []
    for row in completed:
        print(f"Evaluating {row['arm']} seed {row['seed']}: {row['run_dir']}", flush=True)
        per_episode.extend(evaluate_run(row, args.episodes, args.seed, args.deterministic))

    run_summaries = summarize(per_episode)
    arms = arm_summary(run_summaries)

    write_csv(output_dir / "per_episode.csv", per_episode)
    write_csv(output_dir / "seed_summary.csv", run_summaries)
    write_csv(output_dir / "arm_summary.csv", arms)
    write_report(
        output_dir / "after_action_report.md",
        manifest_path,
        completed,
        missing,
        run_summaries,
        arms,
        args,
    )

    print(f"Wrote report files to {output_dir}")


if __name__ == "__main__":
    main()
