"""Run alanine training and plot running best energy vs cumulative evaluations.

This study trains the alanine PPO agent via run_training(...), then uses the
per-method energy-evaluation traces emitted by train.py to compare sample
efficiency across methods. For RL methods, evaluations count visited states
during training. For GD/Adam, evaluations count descent steps on the matched
episode starts used by the training baselines.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .config import TrainConfig
from .train import run_training

METHOD_ORDER = (
    "gd",
    "adam",
    "rl_no_oracle",
    "rl_visible_oracle",
    "rl_hidden_gradient",
)

METHOD_LABELS = {
    "gd": "GD",
    "adam": "Adam",
    "rl_no_oracle": "RL no oracle",
    "rl_visible_oracle": "RL visible oracle",
    "rl_hidden_gradient": "RL hidden gradient",
}


def _parse_int_list(raw: str, arg_name: str, *, min_value: int | None = None) -> list[int]:
    values: list[int] = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        value = int(token)
        if min_value is not None and value < min_value:
            raise ValueError(f"{arg_name} must be >= {min_value}, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must include at least one integer")
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _stats(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "ci95": None,
            "num_values": 0,
        }
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(float(finite.size))) if finite.size > 1 else 0.0
    return {
        "mean": float(np.mean(finite)),
        "std": std,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
        "ci95": ci95,
        "num_values": int(finite.size),
    }


def _method_colors() -> dict[str, tuple]:
    palette = plt.cm.plasma(np.linspace(0.12, 0.88, max(1, len(METHOD_ORDER))))
    return {method: palette[idx] for idx, method in enumerate(METHOD_ORDER)}


def _make_axes(figsize: tuple[float, float] = (8.8, 5.2)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.minorticks_on()
    return fig, ax


def _stepwise_resample(
    evaluations: np.ndarray,
    running_best_energy: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    evals = np.asarray(evaluations, dtype=np.int64).reshape(-1)
    best = np.asarray(running_best_energy, dtype=np.float64).reshape(-1)
    if evals.size == 0 or best.size == 0:
        return np.full(grid.shape, np.nan, dtype=np.float64)
    order = np.argsort(evals)
    evals = evals[order]
    best = best[order]
    indices = np.searchsorted(evals, grid, side="right") - 1
    sampled = np.full(grid.shape, np.nan, dtype=np.float64)
    valid = indices >= 0
    sampled[valid] = best[indices[valid]]
    return sampled


def _build_train_config(args: argparse.Namespace, seed: int, run_name: str) -> TrainConfig:
    return TrainConfig(
        task="alanine_dipeptide",
        seed=int(seed),
        run_name=str(run_name),
        logdir=str(args.logdir),
        device=str(args.device),
        K_map=int(args.K_map),
        K_relax=int(args.K_relax),
        energy_json=str(args.energy_json),
        use_synthetic_fallback=bool(args.use_synthetic_fallback),
        token_noise_std=float(args.token_noise_std),
        ppo_step_scale=float(args.ppo_step_scale),
        step_size=float(args.step_size),
        success_threshold=float(args.success_threshold),
        policy_arch=str(args.policy_arch),
        enable_baselines=not bool(args.disable_baselines),
        tune_baseline_lrs=not bool(args.disable_baseline_lr_tuning),
        baseline_lr_candidates=str(args.baseline_lr_candidates),
        baseline_lr_tune_tasks=int(args.baseline_lr_tune_tasks),
        enable_optimization_curve_eval=False,
        n_env=int(args.n_env),
        train_steps=int(args.train_steps),
        rollout_len=int(args.rollout_len),
        sensing=str(args.sensing),
        running_avg_window=int(args.running_avg_window),
        save_metrics_interval_episodes=int(args.save_metrics_interval_episodes),
        eval_interval_episodes=int(args.eval_interval_episodes),
        eval_episodes=int(args.eval_episodes),
        max_horizon=int(args.max_horizon),
        lr=float(args.lr),
        lr_scheduler=str(args.lr_scheduler),
        lr_min_factor=float(args.lr_min_factor),
        lr_warmup_updates=int(args.lr_warmup_updates),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_ratio=float(args.clip_ratio),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        max_grad_norm=float(args.max_grad_norm),
        ppo_epochs=int(args.ppo_epochs),
        minibatches=int(args.minibatches),
        hidden_dim=int(args.policy_hidden_dim),
        oracle_proj_dim=int(args.oracle_proj_dim),
        s1_step_penalty=float(args.s1_step_penalty),
        enable_training_plots=False,
        use_simple_s_star=bool(args.simple_s_star),
    )


def _aggregate_seed_traces(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    global_min_energy = None
    methods: dict[str, dict[str, Any]] = {}
    for method_key in METHOD_ORDER:
        per_seed = []
        for payload in seed_payloads:
            if global_min_energy is None and payload.get("global_min_energy") is not None:
                global_min_energy = float(payload["global_min_energy"])
            method_payload = payload.get("methods", {}).get(method_key)
            if method_payload is None:
                continue
            evals = np.asarray(method_payload.get("evaluations", []), dtype=np.int64)
            best = np.asarray(method_payload.get("running_best_energy", []), dtype=np.float64)
            if evals.size == 0 or best.size == 0:
                continue
            per_seed.append({
                "seed": int(payload["seed"]),
                "evaluations": evals,
                "running_best_energy": best,
                "total_evaluations": int(method_payload.get("total_evaluations", int(evals[-1]))),
                "best_energy": float(method_payload.get("best_energy", float(best[-1]))),
            })
        if not per_seed:
            continue

        max_eval = max(int(item["total_evaluations"]) for item in per_seed)
        grid_size = min(max_eval, 1024)
        if grid_size <= 1:
            grid = np.asarray([max_eval], dtype=np.int64)
        else:
            grid = np.unique(np.round(np.linspace(1, max_eval, num=grid_size)).astype(np.int64))
        resampled_rows = np.stack(
            [
                _stepwise_resample(
                    item["evaluations"],
                    item["running_best_energy"],
                    grid,
                )
                for item in per_seed
            ],
            axis=0,
        )
        mean_curve = np.nanmean(resampled_rows, axis=0)
        std_curve = np.nanstd(resampled_rows, axis=0)
        ci95_curve = np.zeros_like(mean_curve)
        for idx in range(resampled_rows.shape[1]):
            column = resampled_rows[:, idx]
            finite = column[np.isfinite(column)]
            if finite.size <= 1:
                continue
            ci95_curve[idx] = float(
                1.96 * np.std(finite, ddof=1) / np.sqrt(float(finite.size))
            )

        methods[method_key] = {
            "label": METHOD_LABELS.get(method_key, method_key),
            "evaluation_grid": [int(v) for v in grid],
            "mean_running_best_energy": [float(v) for v in mean_curve],
            "std_running_best_energy": [float(v) for v in std_curve],
            "ci95_running_best_energy": [float(v) for v in ci95_curve],
            "seed_best_energy_values": [float(item["best_energy"]) for item in per_seed],
            "seed_total_evaluations": [int(item["total_evaluations"]) for item in per_seed],
            "seed_best_energy_stats": _stats([float(item["best_energy"]) for item in per_seed]),
            "seed_total_evaluations_stats": _stats(
                [float(item["total_evaluations"]) for item in per_seed]
            ),
        }

    return {
        "method_order": [method_key for method_key in METHOD_ORDER if method_key in methods],
        "energy_units": "kJ/mol",
        "global_min_energy": (float(global_min_energy) if global_min_energy is not None else None),
        "num_seed_payloads": int(len(seed_payloads)),
        "methods": methods,
    }


def _plot_running_best_energy_vs_evaluations(
    *,
    aggregate: dict[str, Any],
    output_path: Path,
) -> None:
    methods = aggregate.get("methods", {})
    if not methods:
        return
    colors = _method_colors()
    fig, ax = _make_axes()
    finite_chunks: list[np.ndarray] = []
    for method_key in METHOD_ORDER:
        method_payload = methods.get(method_key)
        if method_payload is None:
            continue
        grid = np.asarray(method_payload.get("evaluation_grid", []), dtype=np.int64)
        mean_curve = np.asarray(method_payload.get("mean_running_best_energy", []), dtype=np.float64)
        ci95_curve = np.asarray(method_payload.get("ci95_running_best_energy", []), dtype=np.float64)
        if grid.size == 0 or mean_curve.size == 0:
            continue
        color = colors[method_key]
        label = METHOD_LABELS.get(method_key, method_key)
        ax.plot(grid, mean_curve, color=color, linewidth=2.0, label=label)
        if ci95_curve.size == mean_curve.size:
            ax.fill_between(
                grid,
                mean_curve - ci95_curve,
                mean_curve + ci95_curve,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
        finite = mean_curve[np.isfinite(mean_curve)]
        if finite.size > 0:
            finite_chunks.append(finite)

    global_min_energy = aggregate.get("global_min_energy")
    if global_min_energy is not None and np.isfinite(float(global_min_energy)):
        ax.axhline(
            float(global_min_energy),
            color="#57606a",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
            label="True global minimum",
        )

    ax.set_title(
        "Alanine dipeptide | running best energy vs cumulative evaluations",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Energy-surface evaluations")
    ax.set_ylabel("Running best energy (kJ/mol)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if finite_chunks:
        finite_values = np.concatenate(finite_chunks)
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        margin = 0.08 * max(1e-6, y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Alanine experiment: running best energy vs cumulative evaluations."
    )
    parser.add_argument("--seeds", type=str, default=str(defaults.seed))
    parser.add_argument("--study_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="plots/alanine_energy_eval_experiment")
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--K_map", type=int, default=defaults.K_map)
    parser.add_argument("--K_relax", type=int, default=defaults.K_relax)
    parser.add_argument("--energy_json", type=str, default=defaults.energy_json)
    parser.add_argument("--use_synthetic_fallback", action="store_true")
    parser.add_argument("--train_steps", type=int, default=defaults.train_steps)
    parser.add_argument("--n_env", type=int, default=defaults.n_env)
    parser.add_argument("--rollout_len", type=int, default=defaults.rollout_len)
    parser.add_argument("--max_horizon", type=int, default=defaults.max_horizon)
    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=defaults.sensing)
    parser.add_argument("--token_noise_std", type=float, default=defaults.token_noise_std)
    parser.add_argument("--ppo_step_scale", type=float, default=defaults.ppo_step_scale)
    parser.add_argument("--step_size", type=float, default=defaults.step_size)
    parser.add_argument("--success_threshold", type=float, default=defaults.success_threshold)
    parser.add_argument("--policy_arch", type=str, choices=["mlp", "gru"], default=defaults.policy_arch)
    parser.add_argument("--disable_baselines", action="store_true")
    parser.add_argument("--disable_baseline_lr_tuning", action="store_true")
    parser.add_argument("--baseline_lr_candidates", type=str, default=defaults.baseline_lr_candidates)
    parser.add_argument("--baseline_lr_tune_tasks", type=int, default=defaults.baseline_lr_tune_tasks)
    parser.add_argument("--running_avg_window", type=int, default=defaults.running_avg_window)
    parser.add_argument(
        "--save_metrics_interval_episodes",
        type=int,
        default=defaults.save_metrics_interval_episodes,
    )
    parser.add_argument("--eval_interval_episodes", type=int, default=defaults.eval_interval_episodes)
    parser.add_argument("--eval_episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["none", "constant", "linear", "cosine"],
        default=defaults.lr_scheduler,
    )
    parser.add_argument("--lr_min_factor", type=float, default=defaults.lr_min_factor)
    parser.add_argument("--lr_warmup_updates", type=int, default=defaults.lr_warmup_updates)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae_lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip_ratio", type=float, default=defaults.clip_ratio)
    parser.add_argument("--entropy_coef", type=float, default=defaults.entropy_coef)
    parser.add_argument("--value_coef", type=float, default=defaults.value_coef)
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument("--policy_hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--oracle_proj_dim", type=int, default=defaults.oracle_proj_dim)
    parser.add_argument("--s1_step_penalty", type=float, default=defaults.s1_step_penalty)
    parser.add_argument("--simple_s_star", action="store_true")
    parser.add_argument("--skip_plotting", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(str(args.seeds), "seeds")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    study_name = str(args.study_name).strip() or (
        f"alanine_energy_eval_Kmap{int(args.K_map)}_Krelax{int(args.K_relax)}_{timestamp}"
    )
    study_root = Path(str(args.output_dir)).expanduser().resolve() / study_name
    plots_root = study_root / "plots"
    data_root = study_root / "plot_data"
    plots_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    seed_payloads: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

    for seed in seeds:
        run_name = f"alanine_energy_eval_seed{int(seed)}"
        config = _build_train_config(args, seed=int(seed), run_name=run_name)
        print(
            f"[alanine_energy_eval] seed={int(seed)} run={run_name} "
            f"(train_steps={int(config.train_steps)}, n_env={int(config.n_env)})"
        )
        output = run_training(config, return_artifacts=False)
        trace_payload = dict(output.get("energy_evaluation_traces", {}))
        trace_payload["seed"] = int(seed)
        trace_path = data_root / f"seed{int(seed)}_energy_evaluation_trace.json"
        with trace_path.open("w", encoding="utf-8") as handle:
            json.dump(trace_payload, handle, indent=2)
        seed_payloads.append(trace_payload)
        run_entries.append(
            {
                "seed": int(seed),
                "run_dir": str(output["summary"]["run_dir"]),
                "summary_json": str(Path(str(output["summary"]["run_dir"])) / "summary.json"),
                "trace_json": str(trace_path),
            }
        )

    aggregate = _aggregate_seed_traces(seed_payloads)
    aggregate_path = data_root / "aggregate_energy_evaluation.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    plot_paths: dict[str, str] = {}
    if not bool(args.skip_plotting):
        curve_plot = plots_root / "running_best_energy_vs_evaluations.png"
        _plot_running_best_energy_vs_evaluations(aggregate=aggregate, output_path=curve_plot)
        if curve_plot.exists():
            plot_paths["running_best_energy_vs_evaluations"] = str(curve_plot)

    summary = {
        "study_root": str(study_root),
        "aggregate_json": str(aggregate_path),
        "plot_paths": plot_paths,
        "run_entries": run_entries,
    }
    with (study_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[alanine_energy_eval] study_root={study_root}")
    if plot_paths:
        for key, value in plot_paths.items():
            print(f"[alanine_energy_eval] {key}: {value}")


if __name__ == "__main__":
    main()
