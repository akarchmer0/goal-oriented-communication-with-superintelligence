"""Sweep K_relax and K_map for SDP vs norm-ball oracle on alanine dipeptide.

Two independent sweeps (not a full grid):
  1. K_relax sweep: K_relax in [2,4,6,8], K_map=16 (default)
  2. K_map sweep:   K_map in [4,8,16],   K_relax=8 (default)

For each config, trains RL hidden-gradient PPO (no parallel baselines) under
both SDP and norm-ball oracle, evaluates on held-out tasks, and generates
viridis-coded optimization-step plots.

Example:
    python -m tasks.alanine_dipeptide.run_oracle_sweep \
        --seeds 0,1,2 --train_steps 300000
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from .config import TrainConfig
from .train import (
    _resolve_device,
    _rollout_adam_curve,
    _rollout_descent_curve,
    _rollout_policy_curve,
    run_training,
)

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
    seen: set[int] = set()
    return [v for v in values if not (v in seen or seen.add(v))]


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "ci95": float("nan"), "num_values": 0}
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(float(finite.size))) if finite.size > 1 else 0.0
    return {"mean": float(np.mean(finite)), "std": std, "median": float(np.median(finite)),
            "ci95": ci, "num_values": int(finite.size)}


def _make_axes(figsize=(9.5, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.minorticks_on()
    return fig, ax


# ---------------------------------------------------------------------------
# Post-training evaluation (hidden-gradient RL + Adam baseline only)
# ---------------------------------------------------------------------------

def _evaluate_trained_model(
    *,
    seed: int,
    device: torch.device,
    model: torch.nn.Module,
    env,
    num_tasks: int,
    horizon: int,
) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()

    rl_curves: list[np.ndarray] = []
    adam_curves: list[np.ndarray] = []
    gd_curves: list[np.ndarray] = []

    for _ in range(max(1, num_tasks)):
        start_xy = env.rng.uniform(0.0, TWO_PI, size=int(env.visible_dim)).astype(np.float32)
        rl_curves.append(_rollout_policy_curve(
            model=model, env=env, device=device,
            start_xy=start_xy, horizon=horizon, env_index=0,
        ))
        adam_curves.append(_rollout_adam_curve(
            env=env, start_xy=start_xy, horizon=horizon,
            env_index=0, base_lr=env.baseline_lr_adam,
        ))
        gd_curves.append(_rollout_descent_curve(
            env=env, start_xy=start_xy, horizon=horizon,
            env_index=0, base_lr=env.baseline_lr_gd,
        ))

    if was_training:
        model.train()

    threshold = 0.01
    result = {}
    for name, curves_list in [("rl_hidden_gradient", rl_curves), ("adam", adam_curves), ("gd", gd_curves)]:
        curves = np.stack(curves_list, axis=0).astype(np.float64)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        final_vals = curves[:, -1]
        best_vals = np.nanmin(curves, axis=1)
        result[name] = {
            "mean_curve": mean_curve.tolist(),
            "std_curve": std_curve.tolist(),
            "final_objective_stats": _stats([float(v) for v in final_vals]),
            "best_objective_stats": _stats([float(v) for v in best_vals]),
            "final_success_rate": float(np.mean(final_vals <= threshold)),
            "ever_success_rate": float(np.mean(best_vals <= threshold)),
        }

    return {"seed": seed, "num_tasks": num_tasks, "horizon": horizon, "methods": result}


# ---------------------------------------------------------------------------
# Learning curve extraction
# ---------------------------------------------------------------------------

def _extract_learning_curves(run_dir: Path, window: int = 50) -> dict[str, Any]:
    jsonl_path = run_dir / "metrics.jsonl"
    if not jsonl_path.exists():
        return {}
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}

    episodes = [int(r["episodes"]) for r in rows]
    steps = [int(r["steps"]) for r in rows]
    success_raw = [float(r.get("success", 0.0)) for r in rows]
    success_rate = [float(r.get("success_rate", float("nan"))) for r in rows]
    avg_objective = [float(r.get("avg_final_objective", float("nan"))) for r in rows]

    # Smooth success with custom window
    buf: list[float] = []
    success_smoothed: list[float] = []
    for v in success_raw:
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        success_smoothed.append(float(np.mean(buf)))

    return {
        "episodes": episodes, "steps": steps,
        "success_rate": success_rate,
        "success_smoothed": success_smoothed,
        "avg_final_objective": avg_objective,
        "num_episodes": len(rows),
    }


# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------

def _build_config(
    args: argparse.Namespace,
    *,
    seed: int,
    K_map: int,
    K_relax: int,
    use_simple_s_star: bool,
    run_name: str,
) -> TrainConfig:
    return TrainConfig(
        task="alanine_dipeptide",
        seed=seed,
        run_name=run_name,
        logdir=str(args.logdir),
        device=str(args.device),
        K_map=K_map,
        K_relax=K_relax,
        energy_json=str(args.energy_json),
        use_synthetic_fallback=bool(args.use_synthetic_fallback),
        train_steps=int(args.train_steps),
        n_env=int(args.n_env),
        rollout_len=int(args.rollout_len),
        running_avg_window=int(args.running_avg_window),
        save_metrics_interval_episodes=int(args.save_metrics_interval_episodes),
        eval_interval_episodes=int(args.eval_interval_episodes),
        max_horizon=int(args.max_horizon),
        sensing="S0",
        oracle_mode="convex_gradient",
        lr=float(args.lr),
        ppo_epochs=int(args.ppo_epochs),
        minibatches=int(args.minibatches),
        hidden_dim=int(args.policy_hidden_dim),
        oracle_proj_dim=int(args.oracle_proj_dim),
        step_size=float(args.step_size),
        ppo_step_scale=float(args.ppo_step_scale),
        success_threshold=float(args.success_threshold),
        policy_arch=str(args.policy_arch),
        enable_baselines=False,
        tune_baseline_lrs=True,
        baseline_lr_candidates=str(args.baseline_lr_candidates),
        baseline_lr_tune_tasks=int(args.baseline_lr_tune_tasks),
        enable_optimization_curve_eval=False,
        enable_training_plots=False,
        use_simple_s_star=use_simple_s_star,
    )


# ---------------------------------------------------------------------------
# Single config runner (across seeds)
# ---------------------------------------------------------------------------

def _run_config(
    args: argparse.Namespace,
    *,
    K_map: int,
    K_relax: int,
    oracle_type: str,  # "sdp" or "norm_ball"
    seeds: list[int],
    data_root: Path,
) -> dict[str, Any]:
    use_simple = oracle_type == "norm_ball"
    config_key = f"{oracle_type}_Km{K_map}_Kr{K_relax}"
    device = _resolve_device(str(args.device))

    seed_evals: list[dict] = []
    seed_lcs: list[dict] = []
    seed_summaries: list[dict] = []

    for seed in seeds:
        run_name = f"sweep_{config_key}_seed{seed}"
        config = _build_config(
            args, seed=seed, K_map=K_map, K_relax=K_relax,
            use_simple_s_star=use_simple, run_name=run_name,
        )
        print(f"\n[sweep] {config_key} seed={seed} (D={config.spatial_hidden_dim}, "
              f"train_steps={config.train_steps})")

        t0 = time.perf_counter()
        output = run_training(config, return_artifacts=True)
        wall_sec = time.perf_counter() - t0

        run_dir = Path(str(output["summary"]["run_dir"])).expanduser().resolve()

        # Learning curves
        lc = _extract_learning_curves(run_dir, window=int(args.running_avg_window))
        lc["seed"] = seed
        seed_lcs.append(lc)

        # Evaluation
        ev = _evaluate_trained_model(
            seed=seed, device=device,
            model=output["model"], env=output["env"],
            num_tasks=int(args.eval_num_tasks), horizon=int(args.eval_horizon),
        )
        seed_evals.append(ev)

        # SDP info
        sdp_info = {}
        sdp_path = run_dir / "sdp_info.json"
        if sdp_path.exists():
            with sdp_path.open() as f:
                sdp_info = json.load(f)

        seed_summaries.append({
            "seed": seed, "run_dir": str(run_dir),
            "wall_time_sec": wall_sec,
            "summary": output["summary"],
            "sdp_info": sdp_info,
        })
        del output

    # Aggregate eval across seeds
    agg_rl_final = [ev["methods"]["rl_hidden_gradient"]["final_objective_stats"]["mean"]
                    for ev in seed_evals]
    agg_rl_success = [ev["methods"]["rl_hidden_gradient"]["ever_success_rate"]
                      for ev in seed_evals]
    # Aggregate mean curves across seeds
    rl_mean_curves = [np.asarray(ev["methods"]["rl_hidden_gradient"]["mean_curve"])
                      for ev in seed_evals]
    if rl_mean_curves:
        stacked = np.stack(rl_mean_curves)
        grand_mean = np.mean(stacked, axis=0).tolist()
        grand_std = np.std(stacked, axis=0).tolist()
    else:
        grand_mean, grand_std = [], []

    # Same for adam
    adam_mean_curves = [np.asarray(ev["methods"]["adam"]["mean_curve"]) for ev in seed_evals]
    if adam_mean_curves:
        adam_grand_mean = np.mean(np.stack(adam_mean_curves), axis=0).tolist()
    else:
        adam_grand_mean = []

    aggregate = {
        "config_key": config_key,
        "oracle_type": oracle_type,
        "K_map": K_map,
        "K_relax": K_relax,
        "D": int(seed_summaries[0]["summary"].get("hidden_dim", 0)) if seed_summaries else 0,
        "num_seeds": len(seeds),
        "rl_final_objective_stats": _stats(agg_rl_final),
        "rl_ever_success_rate_stats": _stats(agg_rl_success),
        "rl_grand_mean_curve": grand_mean,
        "rl_grand_std_curve": grand_std,
        "adam_grand_mean_curve": adam_grand_mean,
        "seed_summaries": seed_summaries,
        "seed_evals": seed_evals,
        "seed_learning_curves": seed_lcs,
    }

    agg_path = data_root / f"{config_key}_aggregate.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_sweep_curves(
    *,
    configs: list[dict],
    sweep_param: str,
    title: str,
    output_path: Path,
    include_adam: bool = True,
):
    """Plot optimization curves for a set of configs, viridis-coded by sweep_param value."""
    fig, ax = _make_axes(figsize=(10, 6))

    # Assign viridis colors based on parameter values
    param_vals = sorted(set(c[sweep_param] for c in configs))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(param_vals) - 0.5, vmax=max(param_vals) + 0.5)

    for cfg in configs:
        mean_curve = np.asarray(cfg.get("rl_grand_mean_curve", []))
        std_curve = np.asarray(cfg.get("rl_grand_std_curve", []))
        if mean_curve.size < 2:
            continue

        pval = cfg[sweep_param]
        color = cmap(norm(pval))
        oracle = cfg["oracle_type"]
        linestyle = "-" if oracle == "sdp" else "--"
        label = f"{oracle.upper()} {sweep_param}={pval}"
        if oracle == "norm_ball":
            label = f"Norm-ball {sweep_param}={pval}"

        steps = np.arange(mean_curve.size)
        ax.plot(steps, mean_curve, color=color, linewidth=2.0, linestyle=linestyle, label=label)
        if std_curve.size == mean_curve.size:
            ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                            color=color, alpha=0.10, linewidth=0)

    # Adam reference from the first config
    if include_adam and configs:
        adam_curve = np.asarray(configs[0].get("adam_grand_mean_curve", []))
        if adam_curve.size >= 2:
            ax.plot(np.arange(adam_curve.size), adam_curve, color="#9ca3af",
                    linewidth=1.5, linestyle=":", label="Adam", alpha=0.7)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Normalized objective")
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_sweep_bar(
    *,
    configs: list[dict],
    sweep_param: str,
    title: str,
    output_path: Path,
    metric: str = "rl_final_objective_stats",
    ylabel: str = "Final objective",
):
    """Bar chart of a metric across sweep configs, viridis-coded."""
    fig, ax = _make_axes(figsize=(10, 5.5))

    param_vals = sorted(set(c[sweep_param] for c in configs))
    cmap = plt.cm.viridis
    norm_c = plt.Normalize(vmin=min(param_vals) - 0.5, vmax=max(param_vals) + 0.5)

    labels = []
    means = []
    cis = []
    colors = []
    for cfg in sorted(configs, key=lambda c: (c["oracle_type"], c[sweep_param])):
        oracle = cfg["oracle_type"]
        pval = cfg[sweep_param]
        stats = cfg.get(metric, {})
        m = float(stats.get("mean", float("nan")))
        ci = float(stats.get("ci95", 0.0))
        lbl = f"{'SDP' if oracle == 'sdp' else 'NB'} {sweep_param}={pval}"
        labels.append(lbl)
        means.append(m)
        cis.append(ci if np.isfinite(ci) else 0.0)
        colors.append(cmap(norm_c(pval)))

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, color=colors, alpha=0.85, edgecolor="white",
           linewidth=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_learning_curves_sweep(
    *,
    configs: list[dict],
    sweep_param: str,
    title: str,
    output_path: Path,
):
    """Training success-rate learning curves, viridis-coded by sweep param."""
    fig, ax = _make_axes(figsize=(10, 6))

    param_vals = sorted(set(c[sweep_param] for c in configs))
    cmap = plt.cm.viridis
    norm_c = plt.Normalize(vmin=min(param_vals) - 0.5, vmax=max(param_vals) + 0.5)

    for cfg in configs:
        pval = cfg[sweep_param]
        oracle = cfg["oracle_type"]
        color = cmap(norm_c(pval))
        linestyle = "-" if oracle == "sdp" else "--"
        label = f"{'SDP' if oracle == 'sdp' else 'NB'} {sweep_param}={pval}"

        lcs = cfg.get("seed_learning_curves", [])
        if not lcs:
            continue

        # Interpolate to common episode grid
        all_eps = []
        all_sr = []
        for lc in lcs:
            if not lc or "episodes" not in lc:
                continue
            all_eps.append(np.array(lc["episodes"], dtype=np.float64))
            all_sr.append(np.array(lc["success_smoothed"], dtype=np.float64))

        if not all_eps:
            continue

        max_ep = max(float(ep[-1]) for ep in all_eps)
        common_eps = np.linspace(0, max_ep, 500)
        interp = np.array([np.interp(common_eps, ep, sr) for ep, sr in zip(all_eps, all_sr)])
        mean_sr = np.mean(interp, axis=0)
        std_sr = np.std(interp, axis=0)

        ax.plot(common_eps, mean_sr * 100, color=color, linewidth=2.0,
                linestyle=linestyle, label=label)
        ax.fill_between(common_eps, (mean_sr - std_sr) * 100, (mean_sr + std_sr) * 100,
                        color=color, alpha=0.10, linewidth=0)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.set_ylim(-2, 102)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Sweep K_relax and K_map for SDP vs norm-ball oracle."
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--suite_output_dir", type=str, default="plots/oracle_sweep")
    parser.add_argument("--skip_plotting", action="store_true")

    # Sweep ranges
    parser.add_argument("--K_relax_values", type=str, default="2,4,6,8",
                        help="Comma-separated K_relax values to sweep.")
    parser.add_argument("--K_map_values", type=str, default="4,8,16",
                        help="Comma-separated K_map values to sweep.")
    parser.add_argument("--default_K_relax", type=int, default=8,
                        help="K_relax used when sweeping K_map.")
    parser.add_argument("--default_K_map", type=int, default=16,
                        help="K_map used when sweeping K_relax.")

    # Training config
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--energy_json", type=str, default=defaults.energy_json)
    parser.add_argument("--use_synthetic_fallback", action="store_true")
    parser.add_argument("--train_steps", type=int, default=defaults.train_steps)
    parser.add_argument("--n_env", type=int, default=defaults.n_env)
    parser.add_argument("--rollout_len", type=int, default=defaults.rollout_len)
    parser.add_argument("--running_avg_window", type=int, default=defaults.running_avg_window)
    parser.add_argument("--save_metrics_interval_episodes", type=int,
                        default=defaults.save_metrics_interval_episodes)
    parser.add_argument("--eval_interval_episodes", type=int,
                        default=defaults.eval_interval_episodes)
    parser.add_argument("--max_horizon", type=int, default=defaults.max_horizon)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument("--policy_hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--oracle_proj_dim", type=int, default=defaults.oracle_proj_dim)
    parser.add_argument("--step_size", type=float, default=defaults.step_size)
    parser.add_argument("--ppo_step_scale", type=float, default=defaults.ppo_step_scale)
    parser.add_argument("--success_threshold", type=float, default=defaults.success_threshold)
    parser.add_argument("--policy_arch", type=str, choices=["mlp", "gru"],
                        default=defaults.policy_arch)
    parser.add_argument("--disable_baseline_lr_tuning", action="store_true")
    parser.add_argument("--baseline_lr_candidates", type=str,
                        default=defaults.baseline_lr_candidates)
    parser.add_argument("--baseline_lr_tune_tasks", type=int,
                        default=defaults.baseline_lr_tune_tasks)

    # Eval
    parser.add_argument("--eval_num_tasks", type=int, default=500)
    parser.add_argument("--eval_horizon", type=int, default=defaults.max_horizon)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds, "seeds", min_value=0)
    K_relax_values = _parse_int_list(args.K_relax_values, "K_relax_values", min_value=1)
    K_map_values = _parse_int_list(args.K_map_values, "K_map_values", min_value=1)
    default_K_map = int(args.default_K_map)
    default_K_relax = int(args.default_K_relax)

    suite_root = Path(args.suite_output_dir).expanduser().resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    data_root = suite_root / "plot_data"
    plots_root = suite_root / "plots"
    data_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    # Build unique config list: (K_map, K_relax, oracle_type, sweep_group)
    # Deduplicate so we don't train the same config twice.
    # For norm_ball, K_relax is irrelevant, so we key by (K_map, oracle_type) only.
    configs_to_run: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int, str]] = set()
    seen_nb: set[int] = set()  # norm_ball dedup by K_map only

    # K_relax sweep (fixed K_map = default_K_map)
    for kr in K_relax_values:
        # SDP: each K_relax is different
        key = (default_K_map, kr, "sdp")
        if key not in seen:
            seen.add(key)
            configs_to_run.append((default_K_map, kr, "sdp", "K_relax_sweep"))
        # Norm-ball: only need one per K_map
        if default_K_map not in seen_nb:
            seen_nb.add(default_K_map)
            configs_to_run.append((default_K_map, kr, "norm_ball", "K_relax_sweep"))

    # K_map sweep (fixed K_relax = default_K_relax)
    for km in K_map_values:
        for otype in ("sdp", "norm_ball"):
            if otype == "norm_ball":
                if km in seen_nb:
                    continue
                seen_nb.add(km)
            key = (km, default_K_relax, otype)
            if key not in seen:
                seen.add(key)
                configs_to_run.append((km, default_K_relax, otype, "K_map_sweep"))

    n_total = len(configs_to_run) * len(seeds)
    print(f"Oracle sweep experiment")
    print(f"  {len(configs_to_run)} unique configs × {len(seeds)} seeds = {n_total} training runs")
    print(f"  K_relax sweep: {K_relax_values} (K_map={default_K_map})")
    print(f"  K_map sweep: {K_map_values} (K_relax={default_K_relax})")
    print(f"  Seeds: {seeds}")

    # Run all configs
    all_results: dict[str, dict] = {}  # keyed by config_key
    run_idx = 0

    for K_map, K_relax, oracle_type, sweep_group in configs_to_run:
        config_key = f"{oracle_type}_Km{K_map}_Kr{K_relax}"
        run_idx += 1
        print(f"\n{'='*60}")
        print(f"Config {run_idx}/{len(configs_to_run)}: {config_key} "
              f"(sweep={sweep_group}, {len(seeds)} seeds)")
        print(f"{'='*60}")

        result = _run_config(
            args, K_map=K_map, K_relax=K_relax, oracle_type=oracle_type,
            seeds=seeds, data_root=data_root,
        )
        result["sweep_group"] = sweep_group
        all_results[config_key] = result

    # Save combined results
    combined_path = suite_root / "results.json"
    # Strip large arrays for the top-level file (per-seed data is in individual JSONs)
    combined_slim = {}
    for key, res in all_results.items():
        combined_slim[key] = {
            "oracle_type": res["oracle_type"],
            "K_map": res["K_map"],
            "K_relax": res["K_relax"],
            "D": res["D"],
            "num_seeds": res["num_seeds"],
            "sweep_group": res["sweep_group"],
            "rl_final_objective_stats": res["rl_final_objective_stats"],
            "rl_ever_success_rate_stats": res["rl_ever_success_rate_stats"],
            "seed_summaries": [
                {k: v for k, v in s.items() if k != "sdp_info"}
                for s in res.get("seed_summaries", [])
            ],
        }
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump({
            "experiment": "oracle_sweep",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds,
            "K_relax_values": K_relax_values,
            "K_map_values": K_map_values,
            "default_K_map": default_K_map,
            "default_K_relax": default_K_relax,
            "configs": combined_slim,
        }, f, indent=2)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    if args.skip_plotting:
        print("\nSkipping plots.")
    else:
        print("\nGenerating plots...")

        # --- K_relax sweep: SDP configs ---
        kr_sdp = [all_results[f"sdp_Km{default_K_map}_Kr{kr}"]
                  for kr in K_relax_values
                  if f"sdp_Km{default_K_map}_Kr{kr}" in all_results]
        # Norm-ball reference at default K_map (K_relax doesn't matter; find whichever was run)
        kr_nb_ref = []
        for kr in K_relax_values:
            nb_ref_key = f"norm_ball_Km{default_K_map}_Kr{kr}"
            if nb_ref_key in all_results:
                kr_nb_ref = [all_results[nb_ref_key]]
                break

        if kr_sdp:
            _plot_sweep_curves(
                configs=kr_sdp + kr_nb_ref,
                sweep_param="K_relax",
                title=f"K_relax sweep (K_map={default_K_map}): optimization curves",
                output_path=plots_root / "krelax_sweep_optim_curves.png",
            )
            _plot_sweep_bar(
                configs=kr_sdp + kr_nb_ref,
                sweep_param="K_relax",
                title=f"K_relax sweep (K_map={default_K_map}): final objective",
                output_path=plots_root / "krelax_sweep_final_objective.png",
            )
            _plot_sweep_bar(
                configs=kr_sdp + kr_nb_ref,
                sweep_param="K_relax",
                title=f"K_relax sweep (K_map={default_K_map}): success rate",
                output_path=plots_root / "krelax_sweep_success_rate.png",
                metric="rl_ever_success_rate_stats",
                ylabel="Ever-success rate",
            )
            _plot_learning_curves_sweep(
                configs=kr_sdp + kr_nb_ref,
                sweep_param="K_relax",
                title=f"K_relax sweep (K_map={default_K_map}): training success rate",
                output_path=plots_root / "krelax_sweep_learning_curves.png",
            )

        # --- K_map sweep: SDP and norm-ball configs ---
        km_sdp = [all_results[f"sdp_Km{km}_Kr{default_K_relax}"]
                  for km in K_map_values
                  if f"sdp_Km{km}_Kr{default_K_relax}" in all_results]
        # Collect norm-ball configs for each K_map value (may come from either sweep)
        km_nb = []
        for km in K_map_values:
            # Try direct key first, then scan K_relax sweep keys
            found = False
            nb_key = f"norm_ball_Km{km}_Kr{default_K_relax}"
            if nb_key in all_results:
                km_nb.append(all_results[nb_key])
                found = True
            if not found:
                for kr in K_relax_values:
                    nb_key = f"norm_ball_Km{km}_Kr{kr}"
                    if nb_key in all_results:
                        km_nb.append(all_results[nb_key])
                        break

        if km_sdp or km_nb:
            _plot_sweep_curves(
                configs=km_sdp + km_nb,
                sweep_param="K_map",
                title=f"K_map sweep (K_relax={default_K_relax}): optimization curves",
                output_path=plots_root / "kmap_sweep_optim_curves.png",
            )
            _plot_sweep_bar(
                configs=km_sdp + km_nb,
                sweep_param="K_map",
                title=f"K_map sweep (K_relax={default_K_relax}): final objective",
                output_path=plots_root / "kmap_sweep_final_objective.png",
            )
            _plot_sweep_bar(
                configs=km_sdp + km_nb,
                sweep_param="K_map",
                title=f"K_map sweep (K_relax={default_K_relax}): success rate",
                output_path=plots_root / "kmap_sweep_success_rate.png",
                metric="rl_ever_success_rate_stats",
                ylabel="Ever-success rate",
            )
            _plot_learning_curves_sweep(
                configs=km_sdp + km_nb,
                sweep_param="K_map",
                title=f"K_map sweep (K_relax={default_K_relax}): training success rate",
                output_path=plots_root / "kmap_sweep_learning_curves.png",
            )

        # --- Combined: all configs on one plot ---
        all_configs = list(all_results.values())
        if all_configs:
            # Split into SDP and norm-ball for a combined "all" plot using K_map as color
            _plot_sweep_curves(
                configs=all_configs,
                sweep_param="K_map",
                title="All configs: optimization curves (solid=SDP, dashed=norm-ball)",
                output_path=plots_root / "all_configs_optim_curves.png",
            )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Config':<30s} {'D':>5s} {'Final Obj':>12s} {'±CI95':>8s} {'Success%':>10s}")
    print("-" * 80)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        fo = r["rl_final_objective_stats"]
        sr = r["rl_ever_success_rate_stats"]
        print(f"{key:<30s} {r['D']:>5d} "
              f"{fo.get('mean', float('nan')):>12.4f} "
              f"{fo.get('ci95', 0):>8.4f} "
              f"{sr.get('mean', 0) * 100:>9.1f}%")
    print("=" * 80)

    print(f"\nResults: {combined_path}")
    print(f"Plots:   {plots_root}")


if __name__ == "__main__":
    main()
