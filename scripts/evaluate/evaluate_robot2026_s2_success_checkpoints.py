from __future__ import annotations

import argparse
import csv
import json
import os
import re
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


CHECKPOINT_RE = re.compile(r"model_(\d+)\.zip$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select S2 checkpoints by stair success instead of training reward."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="reports/robot2026_camera_ready/s2_success_checkpoint_diagnostic")
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--coarse-episodes", type=int, default=10)
    parser.add_argument("--refine-episodes", type=int, default=20)
    parser.add_argument("--final-episodes", type=int, default=100)
    parser.add_argument("--coarse-interval", type=int, default=1_000_000)
    parser.add_argument("--refine-window", type=int, default=1_000_000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_config(config_path: Path) -> Any:
    cfg = OmegaConf.load(config_path)
    if "env" not in cfg or "make_kwargs" not in cfg.env:
        raise ValueError(f"Config does not contain env.make_kwargs: {config_path}")
    return cfg


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    if not match:
        raise ValueError(f"Not a model checkpoint: {path}")
    return int(match.group(1))


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path, Path]]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    checkpoints: list[tuple[int, Path, Path]] = []
    for model_path in checkpoint_dir.glob("model_*.zip"):
        step = checkpoint_step(model_path)
        vecnorm_path = checkpoint_dir / f"vecnormalize_{step}.pkl"
        if vecnorm_path.is_file():
            checkpoints.append((step, model_path, vecnorm_path))
    return sorted(checkpoints, key=lambda item: item[0])


def select_coarse(checkpoints: list[tuple[int, Path, Path]], interval: int) -> list[tuple[int, Path, Path]]:
    selected = [item for item in checkpoints if item[0] % interval == 0]
    if checkpoints and checkpoints[-1] not in selected:
        selected.append(checkpoints[-1])
    return sorted(set(selected), key=lambda item: item[0])


def select_refine(
    checkpoints: list[tuple[int, Path, Path]], center_step: int, window: int
) -> list[tuple[int, Path, Path]]:
    low = center_step - window
    high = center_step + window
    return [item for item in checkpoints if low <= item[0] <= high]


def build_env(cfg: Any, vecnorm_path: Path, seed: int) -> DummyVecEnv:
    make_kwargs = dict(OmegaConf.to_container(cfg.env.make_kwargs, resolve=True) or {})
    make_kwargs["render_mode"] = None
    env_fn = make_single_env(str(cfg.env.name), make_kwargs, monitor=False, seed=seed)
    venv = DummyVecEnv([env_fn])
    return maybe_load_vecnormalize(venv, str(vecnorm_path))


def evaluate_model(
    cfg: Any,
    model_path: Path,
    vecnorm_path: Path,
    episodes: int,
    eval_seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    make_kwargs = dict(OmegaConf.to_container(cfg.env.make_kwargs, resolve=True) or {})
    target_steps = int(make_kwargs.get("num_steps", 0))
    venv = build_env(cfg, vecnorm_path, eval_seed)
    model = PPO.load(str(model_path))

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    max_steps: list[int] = []
    final_xs: list[float] = []

    try:
        for _ in range(episodes):
            obs = venv.reset()
            done = False
            total_reward = 0.0
            length = 0
            max_step_reached = 0
            final_x = float("nan")

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, infos = venv.step(action)
                info = dict(infos[0])
                total_reward += float(reward[0])
                length += 1
                done = bool(dones[0])
                max_step_reached = max(max_step_reached, int(info.get("max_step_reached", 0)))
                final_x = float(info.get("x_position", np.nan))

            rewards.append(total_reward)
            lengths.append(length)
            successes.append(bool(target_steps and max_step_reached >= target_steps))
            max_steps.append(max_step_reached)
            final_xs.append(final_x)
    finally:
        venv.close()

    return {
        "episodes": episodes,
        "success_rate": 100.0 * sum(successes) / len(successes),
        "mean_reward": statistics.fmean(rewards),
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_length": statistics.fmean(lengths),
        "mean_max_step_reached": statistics.fmean(max_steps),
        "mean_final_x": statistics.fmean(final_xs),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["success_rate"]),
            float(row["mean_max_step_reached"]),
            float(row["mean_final_x"]),
            float(row["mean_reward"]),
            int(row["checkpoint_step"]),
        ),
    )


def summarize_by_arm(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        by_arm[str(row["arm"])].append(row)

    summaries: list[dict[str, Any]] = []
    for arm, rows in sorted(by_arm.items()):
        success_rates = [float(row["success_rate"]) for row in rows]
        rewards = [float(row["mean_reward"]) for row in rows]
        lengths = [float(row["mean_length"]) for row in rows]
        summaries.append(
            {
                "arm": arm,
                "seeds_completed": len(rows),
                "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda item: int(item["seed"]))),
                "mean_success_rate": statistics.fmean(success_rates),
                "sd_success_rate": statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
                "mean_reward_across_seeds": statistics.fmean(rewards),
                "sd_reward_across_seeds": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
                "mean_length_across_seeds": statistics.fmean(lengths),
            }
        )
    return summaries


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.2f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    path: Path,
    args: argparse.Namespace,
    scan_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "eval_seed": args.eval_seed,
        "coarse_episodes": args.coarse_episodes,
        "refine_episodes": args.refine_episodes,
        "final_episodes": args.final_episodes,
        "coarse_interval": args.coarse_interval,
        "refine_window": args.refine_window,
        "scan_checkpoints_evaluated": len(scan_rows),
        "selected_runs": len(selected_rows),
        "missing_runs": len(missing),
    }
    lines = [
        "# S2 Success-Based Checkpoint Diagnostic",
        "",
        "## Protocol",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "This diagnostic does not replace the reward-selected S2 ablation report.",
        "It asks whether stored periodic checkpoints contain better stair-completion policies than the reward-selected `eval/best_model.zip` checkpoints.",
        "",
        "Selection criterion: maximize success rate, then mean maximum step reached, final x-position, mean reward, and later checkpoint step.",
        "",
        "## Final Arm Summary",
        "",
        markdown_table(
            arm_rows,
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
        "## Selected Checkpoints",
        "",
        markdown_table(
            selected_rows,
            [
                "arm",
                "seed",
                "checkpoint_step",
                "selection_stage",
                "selection_success_rate",
                "final_success_rate",
                "final_mean_reward",
                "final_mean_length",
                "final_mean_max_step_reached",
            ],
        ),
        "## Missing Runs",
        "",
    ]
    lines.append(markdown_table(missing, ["arm", "seed", "run_dir", "reason"]) if missing else "_None._\n")
    lines.extend(
        [
            "## Conclusions",
            "",
            "- Fill in after comparing this report with `reports/robot2026_camera_ready/s2_ablation/after_action_report.md`.",
            "- If success-selected checkpoints substantially improve `s2_base`, reward-based checkpoint selection was a major confound.",
            "- If they do not, the five-seed training result should be interpreted as a training instability rather than only a checkpoint-selection issue.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(Path(args.manifest))
    scan_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for row in manifest:
        arm = row["arm"]
        seed = int(row["seed"])
        run_dir = Path(row["run_dir"])
        config_path = Path(row["config_path"])
        if not config_path.is_file():
            missing.append({"arm": arm, "seed": seed, "run_dir": str(run_dir), "reason": "missing config"})
            continue
        checkpoints = list_checkpoints(run_dir)
        if not checkpoints:
            missing.append({"arm": arm, "seed": seed, "run_dir": str(run_dir), "reason": "missing checkpoints"})
            continue
        cfg = load_config(config_path)

        print(f"[coarse] {arm} seed {seed}: {len(checkpoints)} checkpoints", flush=True)
        coarse_rows: list[dict[str, Any]] = []
        for step, model_path, vecnorm_path in select_coarse(checkpoints, args.coarse_interval):
            metrics = evaluate_model(cfg, model_path, vecnorm_path, args.coarse_episodes, args.eval_seed, args.deterministic)
            scan_row = {
                "stage": "coarse",
                "arm": arm,
                "seed": seed,
                "checkpoint_step": step,
                "model_path": str(model_path),
                "vecnorm_path": str(vecnorm_path),
                **metrics,
            }
            coarse_rows.append(scan_row)
            scan_rows.append(scan_row)

        coarse_best = choose_best(coarse_rows)
        print(f"[refine] {arm} seed {seed}: around {coarse_best['checkpoint_step']}", flush=True)
        refine_rows: list[dict[str, Any]] = []
        for step, model_path, vecnorm_path in select_refine(
            checkpoints, int(coarse_best["checkpoint_step"]), args.refine_window
        ):
            metrics = evaluate_model(cfg, model_path, vecnorm_path, args.refine_episodes, args.eval_seed, args.deterministic)
            scan_row = {
                "stage": "refine",
                "arm": arm,
                "seed": seed,
                "checkpoint_step": step,
                "model_path": str(model_path),
                "vecnorm_path": str(vecnorm_path),
                **metrics,
            }
            refine_rows.append(scan_row)
            scan_rows.append(scan_row)

        selected = choose_best(refine_rows)
        print(f"[final] {arm} seed {seed}: checkpoint {selected['checkpoint_step']}", flush=True)
        final_metrics = evaluate_model(
            cfg,
            Path(str(selected["model_path"])),
            Path(str(selected["vecnorm_path"])),
            args.final_episodes,
            args.eval_seed,
            args.deterministic,
        )
        selected_row = {
            "arm": arm,
            "seed": seed,
            "checkpoint_step": int(selected["checkpoint_step"]),
            "selection_stage": str(selected["stage"]),
            "selection_success_rate": float(selected["success_rate"]),
            "selection_mean_reward": float(selected["mean_reward"]),
            "model_path": str(selected["model_path"]),
            "vecnorm_path": str(selected["vecnorm_path"]),
            "final_success_rate": float(final_metrics["success_rate"]),
            "final_mean_reward": float(final_metrics["mean_reward"]),
            "final_std_reward": float(final_metrics["std_reward"]),
            "final_mean_length": float(final_metrics["mean_length"]),
            "final_mean_max_step_reached": float(final_metrics["mean_max_step_reached"]),
            "final_mean_final_x": float(final_metrics["mean_final_x"]),
        }
        selected_rows.append(selected_row)
        final_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "checkpoint_step": int(selected["checkpoint_step"]),
                "episodes": args.final_episodes,
                "success_rate": float(final_metrics["success_rate"]),
                "mean_reward": float(final_metrics["mean_reward"]),
                "std_reward": float(final_metrics["std_reward"]),
                "mean_length": float(final_metrics["mean_length"]),
                "mean_max_step_reached": float(final_metrics["mean_max_step_reached"]),
                "mean_final_x": float(final_metrics["mean_final_x"]),
                "model_path": str(selected["model_path"]),
                "vecnorm_path": str(selected["vecnorm_path"]),
            }
        )

    if args.strict and missing:
        labels = ", ".join(f"{row['arm']} seed {row['seed']}: {row['reason']}" for row in missing)
        raise SystemExit(f"Missing required checkpoint artifacts: {labels}")

    arm_rows = summarize_by_arm(final_rows)
    write_csv(output_dir / "checkpoint_scan_summary.csv", scan_rows)
    write_csv(output_dir / "selected_checkpoints.csv", selected_rows)
    write_csv(output_dir / "final_success_selected_summary.csv", final_rows)
    write_csv(output_dir / "arm_summary.csv", arm_rows)
    write_report(output_dir / "after_action_report.md", args, scan_rows, selected_rows, final_rows, arm_rows, missing)
    print(f"Wrote diagnostic report files to {output_dir}")


if __name__ == "__main__":
    main()
