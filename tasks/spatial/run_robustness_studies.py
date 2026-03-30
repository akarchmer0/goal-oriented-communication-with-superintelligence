"""Robustness and generalization ablation experiments for the spatial task.

Runs three ablation studies:
1. Oracle noise: Train with clean oracle, evaluate under increasing noise levels.
2. Hidden dimension: Train and evaluate with varying basis_complexity (hidden dim).
3. Unseen maps: Train on one Fourier map, evaluate on fresh unseen maps.

Example usage:
    python -m tasks.spatial.run_robustness_studies \
        --seeds 0,1,2 --train_steps 300000 \
        --suite_output_dir plots/spatial_robustness
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

from .config import TrainConfig
from .train import (
    _apply_spatial_task_snapshot,
    _capture_spatial_task_snapshot,
    _resolve_device,
    _rollout_spatial_adam_curve,
    _rollout_spatial_descent_curve,
    _rollout_spatial_policy_curve,
    run_training,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ci95(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 0.0
    return float(1.96 * np.std(finite, ddof=1) / np.sqrt(float(finite.size)))


def _stats(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "ci95": _ci95(finite),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "n": int(finite.size),
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d7de")
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(colors="#57606a")
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)


def _evaluate_rl_hidden(
    *,
    model: torch.nn.Module,
    env,
    device: torch.device,
    num_tasks: int,
    horizon: int,
    seed: int,
) -> list[np.ndarray]:
    """Evaluate RL hidden-gradient agent on num_tasks random tasks. Returns list of curves."""
    model.eval()
    curves: list[np.ndarray] = []
    for _ in range(num_tasks):
        spec = env.sample_episode_spec(env_index=0, refresh_map=env.refresh_map_each_episode)
        snapshot = _capture_spatial_task_snapshot(
            env=env, source_xy=spec.source, target_min_xy=spec.target_min_xy, env_index=0,
        )
        _apply_spatial_task_snapshot(env, snapshot, env_index=0)
        start_xy = snapshot["source_xy"].astype(np.float32)
        curve = _rollout_spatial_policy_curve(
            model=model, env=env, device=device,
            start_xy=start_xy, horizon=horizon, env_index=0,
        )
        curves.append(curve)
    return curves


def _build_config(
    *,
    seed: int,
    run_name: str,
    train_steps: int,
    basis_complexity: int = 3,
    hidden_dim: int = 150,
    max_horizon: int = 60,
    n_env: int = 32,
    device: str = "cpu",
) -> TrainConfig:
    return TrainConfig(
        task="spatial",
        seed=seed,
        run_name=run_name,
        logdir="runs",
        device=device,
        train_steps=train_steps,
        n_env=n_env,
        rollout_len=64,
        running_avg_window=100,
        eval_interval_episodes=200,
        save_metrics_interval_episodes=500,
        max_horizon=max_horizon,
        sensing="S0",
        oracle_mode="convex_gradient",
        lr=3e-4,
        ppo_epochs=4,
        minibatches=4,
        hidden_dim=64,
        oracle_proj_dim=64,
        spatial_hidden_dim=hidden_dim,
        spatial_visible_dim=2,
        spatial_coord_limit=3,
        spatial_step_size=1.0,
        ppo_step_scale=1.0,
        spatial_success_threshold=0.01,
        spatial_enable_success_curriculum=False,
        spatial_basis_complexity=basis_complexity,
        spatial_freq_sparsity=0,
        spatial_f_type="FOURIER",
        spatial_policy_arch="mlp",
        spatial_refresh_map_each_episode=False,
        spatial_fixed_start_target=False,
        spatial_plot_interval_episodes=0,
        spatial_enable_baselines=True,
        spatial_tune_baseline_lrs=True,
        spatial_early_stop_on_all_methods_success=False,
        spatial_baseline_lr_candidates="0.0001,0.0003,0.0007,0.001,0.003,0.007,0.01,0.02,0.03,0.05,",
        spatial_baseline_lr_tune_tasks=300,
        spatial_enable_optimization_curve_eval=False,
        enable_training_plots=False,
        spatial_token_dim=hidden_dim,
    )


# ---------------------------------------------------------------------------
# Study 1: Oracle Noise Ablation
# ---------------------------------------------------------------------------

def run_oracle_noise_study(
    *,
    seeds: list[int],
    train_steps: int,
    num_tasks: int,
    horizon: int,
    noise_levels: list[float],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train once per seed with clean oracle, then evaluate under various noise levels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = _resolve_device(device)
    results: dict[str, Any] = {
        "study": "oracle_noise",
        "noise_levels": noise_levels,
        "seeds": seeds,
        "per_noise_level": {},
    }

    # Collect per-noise-level final objectives across all seeds
    noise_finals: dict[float, list[float]] = {nl: [] for nl in noise_levels}
    noise_curves_all: dict[float, list[np.ndarray]] = {nl: [] for nl in noise_levels}

    for seed in seeds:
        print(f"[oracle_noise] Training seed={seed}")
        config = _build_config(
            seed=seed,
            run_name=f"robustness_oracle_noise_seed{seed}",
            train_steps=train_steps,
            device=device,
        )
        t0 = time.perf_counter()
        output = run_training(config, return_artifacts=True)
        print(f"[oracle_noise] seed={seed} trained in {time.perf_counter() - t0:.1f}s")

        model = output["model"]
        env = output["env"]

        for noise_std in noise_levels:
            # Inject noise into oracle
            original_noise = env.oracle.token_noise_std
            env.oracle.token_noise_std = float(noise_std)

            curves = _evaluate_rl_hidden(
                model=model, env=env, device=device_t,
                num_tasks=num_tasks, horizon=horizon, seed=seed,
            )
            finals = [float(c[-1]) for c in curves]
            noise_finals[noise_std].extend(finals)
            noise_curves_all[noise_std].extend(curves)

            # Restore
            env.oracle.token_noise_std = original_noise
            print(f"  noise_std={noise_std:.2f}: mean_final={np.mean(finals):.4f} +/- {_ci95(np.array(finals)):.4f}")

        del output

    # Aggregate
    for nl in noise_levels:
        arr = np.array(noise_finals[nl])
        curves_stacked = np.stack(noise_curves_all[nl], axis=0)
        mean_curve = np.mean(curves_stacked, axis=0).tolist()
        results["per_noise_level"][str(nl)] = {
            "noise_std": nl,
            **_stats(arr),
            "mean_curve": mean_curve,
        }

    # Save data
    data_path = output_dir / "oracle_noise_results.json"
    with data_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Plot
    _plot_oracle_noise(results, output_dir / "oracle_noise_ablation.png")
    print(f"[oracle_noise] Results saved to {output_dir}")
    return results


def _plot_oracle_noise(results: dict, output_path: Path) -> None:
    noise_levels = results["noise_levels"]
    means = [results["per_noise_level"][str(nl)]["mean"] for nl in noise_levels]
    ci95s = [results["per_noise_level"][str(nl)]["ci95"] for nl in noise_levels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    _style_axis(ax)

    ax.errorbar(
        noise_levels, means, yerr=ci95s,
        fmt="o-", color="#5B21B6", linewidth=2, markersize=6,
        capsize=4, capthick=1.5, ecolor="#8B5CF6",
    )
    ax.set_xlabel("Oracle noise std", fontsize=11)
    ax.set_ylabel("Final normalized objective", fontsize=11)
    ax.set_title("Oracle noise robustness (RL hidden gradient)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Study 2: Hidden Dimension Ablation
# ---------------------------------------------------------------------------

def run_hidden_dim_study(
    *,
    seeds: list[int],
    train_steps: int,
    num_tasks: int,
    horizon: int,
    basis_complexities: list[int],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train and evaluate with different basis_complexity values (which set hidden_dim)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = _resolve_device(device)

    # basis_complexity k -> hidden_dim D: D = visible_dim * (2*k+1)^visible_dim
    # For visible_dim=2: k=1 -> D=2*9=18, k=2 -> D=2*25=50, k=3 -> D=2*49=98...
    # Actually let me check. The hidden_dim is a parameter, not derived from basis_complexity.
    # Looking at the code, hidden_dim is set independently. Let me use the standard mapping.
    # In run_optimizer_studies, spatial_hidden_dim=150 is the default with basis_complexity=3.
    # The hidden_dim controls the Fourier embedding dimension.

    results: dict[str, Any] = {
        "study": "hidden_dim",
        "basis_complexities": basis_complexities,
        "seeds": seeds,
        "per_basis_complexity": {},
    }

    for bc in basis_complexities:
        # Scale hidden_dim proportionally: bc=3 -> D=150 (default)
        hidden_dim = max(10, bc * 50)
        print(f"\n[hidden_dim] basis_complexity={bc}, hidden_dim={hidden_dim}")

        bc_finals: list[float] = []
        bc_curves: list[np.ndarray] = []

        for seed in seeds:
            config = _build_config(
                seed=seed,
                run_name=f"robustness_hdim_bc{bc}_seed{seed}",
                train_steps=train_steps,
                basis_complexity=bc,
                hidden_dim=hidden_dim,
                device=device,
            )
            t0 = time.perf_counter()
            output = run_training(config, return_artifacts=True)
            print(f"  seed={seed} trained in {time.perf_counter() - t0:.1f}s")

            model = output["model"]
            env = output["env"]

            curves = _evaluate_rl_hidden(
                model=model, env=env, device=device_t,
                num_tasks=num_tasks, horizon=horizon, seed=seed,
            )
            finals = [float(c[-1]) for c in curves]
            bc_finals.extend(finals)
            bc_curves.extend(curves)
            print(f"  seed={seed}: mean_final={np.mean(finals):.4f}")
            del output

        arr = np.array(bc_finals)
        curves_stacked = np.stack(bc_curves, axis=0)
        results["per_basis_complexity"][str(bc)] = {
            "basis_complexity": bc,
            "hidden_dim": hidden_dim,
            **_stats(arr),
            "mean_curve": np.mean(curves_stacked, axis=0).tolist(),
        }

    data_path = output_dir / "hidden_dim_results.json"
    with data_path.open("w") as f:
        json.dump(results, f, indent=2)

    _plot_hidden_dim(results, output_dir / "hidden_dim_ablation.png")
    print(f"[hidden_dim] Results saved to {output_dir}")
    return results


def _plot_hidden_dim(results: dict, output_path: Path) -> None:
    bcs = results["basis_complexities"]
    hidden_dims = [results["per_basis_complexity"][str(bc)]["hidden_dim"] for bc in bcs]
    means = [results["per_basis_complexity"][str(bc)]["mean"] for bc in bcs]
    ci95s = [results["per_basis_complexity"][str(bc)]["ci95"] for bc in bcs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    _style_axis(ax)

    ax.errorbar(
        hidden_dims, means, yerr=ci95s,
        fmt="s-", color="#059669", linewidth=2, markersize=7,
        capsize=4, capthick=1.5, ecolor="#34D399",
    )
    ax.set_xlabel("Hidden dimension D", fontsize=11)
    ax.set_ylabel("Final normalized objective", fontsize=11)
    ax.set_title("Hidden dimension ablation (RL hidden gradient)", fontsize=12, fontweight="bold")
    for i, bc in enumerate(bcs):
        ax.annotate(f"k={bc}", (hidden_dims[i], means[i]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Study 3: Generalization to Unseen Maps
# ---------------------------------------------------------------------------

def run_unseen_maps_study(
    *,
    seeds: list[int],
    train_steps: int,
    num_tasks: int,
    horizon: int,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train on one Fourier map per seed, then evaluate on unseen fresh maps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = _resolve_device(device)
    results: dict[str, Any] = {
        "study": "unseen_maps",
        "seeds": seeds,
        "seen_map": {},
        "unseen_maps": {},
    }

    seen_finals: list[float] = []
    unseen_finals: list[float] = []
    seen_curves_all: list[np.ndarray] = []
    unseen_curves_all: list[np.ndarray] = []

    for seed in seeds:
        print(f"[unseen_maps] Training seed={seed} (refresh_map=False)")
        config = _build_config(
            seed=seed,
            run_name=f"robustness_unseen_maps_seed{seed}",
            train_steps=train_steps,
            device=device,
        )
        t0 = time.perf_counter()
        output = run_training(config, return_artifacts=True)
        print(f"  Trained in {time.perf_counter() - t0:.1f}s")

        model = output["model"]
        env = output["env"]

        # Evaluate on the SAME map (training distribution)
        env.refresh_map_each_episode = False
        curves_seen = _evaluate_rl_hidden(
            model=model, env=env, device=device_t,
            num_tasks=num_tasks, horizon=horizon, seed=seed,
        )
        finals_seen = [float(c[-1]) for c in curves_seen]
        seen_finals.extend(finals_seen)
        seen_curves_all.extend(curves_seen)
        print(f"  Same map: mean_final={np.mean(finals_seen):.4f}")

        # Evaluate on UNSEEN maps (fresh Fourier map each task)
        env.refresh_map_each_episode = True
        curves_unseen = _evaluate_rl_hidden(
            model=model, env=env, device=device_t,
            num_tasks=num_tasks, horizon=horizon, seed=seed,
        )
        finals_unseen = [float(c[-1]) for c in curves_unseen]
        unseen_finals.extend(finals_unseen)
        unseen_curves_all.extend(curves_unseen)
        print(f"  Unseen maps: mean_final={np.mean(finals_unseen):.4f}")

        # Reset
        env.refresh_map_each_episode = False
        del output

    results["seen_map"] = _stats(np.array(seen_finals))
    results["seen_map"]["mean_curve"] = np.mean(np.stack(seen_curves_all), axis=0).tolist()
    results["unseen_maps"] = _stats(np.array(unseen_finals))
    results["unseen_maps"]["mean_curve"] = np.mean(np.stack(unseen_curves_all), axis=0).tolist()

    data_path = output_dir / "unseen_maps_results.json"
    with data_path.open("w") as f:
        json.dump(results, f, indent=2)

    _plot_unseen_maps(results, output_dir / "unseen_maps_generalization.png", horizon)
    print(f"[unseen_maps] Results saved to {output_dir}")
    return results


def _plot_unseen_maps(results: dict, output_path: Path, horizon: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    # Left: bar chart of final objective
    _style_axis(ax1)
    labels = ["Same map\n(training)", "Unseen maps\n(fresh Fourier)"]
    means = [results["seen_map"]["mean"], results["unseen_maps"]["mean"]]
    ci95s = [results["seen_map"]["ci95"], results["unseen_maps"]["ci95"]]
    colors = ["#5B21B6", "#DC2626"]
    bars = ax1.bar(labels, means, yerr=ci95s, color=colors, alpha=0.85, capsize=5, width=0.5)
    ax1.set_ylabel("Final normalized objective", fontsize=11)
    ax1.set_title("Generalization: same vs unseen maps", fontsize=12, fontweight="bold")
    for bar, m, ci in zip(bars, means, ci95s):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.005,
                 f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    # Right: mean curves
    _style_axis(ax2)
    steps = np.arange(horizon + 1)
    seen_curve = np.array(results["seen_map"]["mean_curve"])
    unseen_curve = np.array(results["unseen_maps"]["mean_curve"])
    ax2.plot(steps, seen_curve, color="#5B21B6", linewidth=2, label="Same map")
    ax2.plot(steps, unseen_curve, color="#DC2626", linewidth=2, linestyle="--", label="Unseen maps")
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Mean normalized objective", fontsize=11)
    ax2.set_title("Optimization curves", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Study 4: Dimension Scaling
# ---------------------------------------------------------------------------

# Per-dimension configuration: hidden_dim D, lattice granularity G, training
# time multiplier.  Uses basis_complexity K=2 and freq_sparsity r=2 (pairwise
# interactions) so that D grows as O(d^2) rather than exponentially.
SCALING_CONFIGS: dict[int, dict[str, float | int]] = {
    2: {"hidden_dim": 50,  "granularity": 20, "train_steps_factor": 1.0},
    3: {"hidden_dim": 100, "granularity": 15, "train_steps_factor": 1.33},
    4: {"hidden_dim": 200, "granularity": 11, "train_steps_factor": 1.67},
    5: {"hidden_dim": 300, "granularity": 9,  "train_steps_factor": 2.0},
    6: {"hidden_dim": 400, "granularity": 7,  "train_steps_factor": 2.5},
}


def _build_config_scaling(
    *,
    seed: int,
    run_name: str,
    train_steps: int,
    visible_dim: int,
    hidden_dim: int,
    basis_complexity: int = 2,
    freq_sparsity: int = 2,
    lattice_rl: bool = True,
    lattice_granularity: int = 20,
    max_horizon: int = 60,
    n_env: int = 32,
    device: str = "cpu",
) -> TrainConfig:
    return TrainConfig(
        task="spatial",
        seed=seed,
        run_name=run_name,
        logdir="runs",
        device=device,
        train_steps=train_steps,
        n_env=n_env,
        rollout_len=64,
        running_avg_window=100,
        eval_interval_episodes=200,
        save_metrics_interval_episodes=500,
        max_horizon=max_horizon,
        sensing="S0",
        oracle_mode="convex_gradient",
        lr=3e-4,
        ppo_epochs=4,
        minibatches=4,
        hidden_dim=64,
        oracle_proj_dim=64,
        spatial_hidden_dim=hidden_dim,
        spatial_visible_dim=visible_dim,
        spatial_coord_limit=3,
        spatial_step_size=1.0,
        ppo_step_scale=1.0,
        spatial_success_threshold=0.01,
        spatial_enable_success_curriculum=False,
        spatial_basis_complexity=basis_complexity,
        spatial_freq_sparsity=freq_sparsity,
        spatial_f_type="FOURIER",
        spatial_policy_arch="mlp",
        spatial_refresh_map_each_episode=False,
        spatial_fixed_start_target=False,
        spatial_plot_interval_episodes=0,
        spatial_enable_baselines=True,
        spatial_tune_baseline_lrs=True,
        spatial_early_stop_on_all_methods_success=False,
        spatial_baseline_lr_candidates="0.0001,0.0003,0.0007,0.001,0.003,0.007,0.01,0.02,0.03,0.05,",
        spatial_baseline_lr_tune_tasks=300,
        spatial_enable_optimization_curve_eval=False,
        enable_training_plots=False,
        spatial_token_dim=hidden_dim,
        lattice_rl=lattice_rl,
        lattice_granularity=lattice_granularity,
    )


def _rollout_random_search_curve_local(
    *,
    env,
    start_xy: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    env_index: int = 0,
) -> np.ndarray:
    """Random search: sample random points, keep the best-so-far."""
    h = max(1, int(horizon))
    best_obj = float(env._normalized_objective_value(start_xy.astype(np.float32), env_index=env_index))
    curve = np.empty(h + 1, dtype=np.float32)
    curve[0] = best_obj
    for step in range(h):
        candidate = rng.uniform(
            low=-float(env.coord_limit),
            high=float(env.coord_limit),
            size=int(env.visible_dim),
        ).astype(np.float32)
        cand_obj = float(env._normalized_objective_value(candidate, env_index=env_index))
        if cand_obj < best_obj:
            best_obj = cand_obj
        curve[step + 1] = best_obj
    return curve


def run_dimension_scaling_study(
    *,
    seeds: list[int],
    train_steps_base: int,
    num_tasks: int,
    horizon: int,
    visible_dims: list[int],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train and evaluate across increasing visible dimensions.

    Uses freq_sparsity=2 (pairwise) and lattice_rl=True to keep complexity
    manageable.  Compares RL hidden gradient against GD, Adam, and random
    search baselines at each dimension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = _resolve_device(device)

    results: dict[str, Any] = {
        "study": "dimension_scaling",
        "visible_dims": visible_dims,
        "seeds": seeds,
        "basis_complexity": 2,
        "freq_sparsity": 2,
        "per_dim": {},
    }

    for d in visible_dims:
        cfg = SCALING_CONFIGS.get(d, {
            "hidden_dim": max(50, d * 60),
            "granularity": max(5, 22 - 3 * d),
            "train_steps_factor": max(1.0, d / 2.0),
        })
        hidden_dim = int(cfg["hidden_dim"])
        granularity = int(cfg["granularity"])
        train_steps = int(train_steps_base * cfg["train_steps_factor"])

        print(f"\n[dim_scaling] d={d}, D={hidden_dim}, G={granularity}, steps={train_steps}")

        dim_results: dict[str, Any] = {
            "visible_dim": d,
            "hidden_dim": hidden_dim,
            "granularity": granularity,
            "train_steps": train_steps,
            "methods": {},
        }

        method_finals: dict[str, list[float]] = {
            "rl_hidden": [], "gd": [], "adam": [], "random_search": [],
        }
        method_curves: dict[str, list[np.ndarray]] = {
            "rl_hidden": [], "gd": [], "adam": [], "random_search": [],
        }

        for seed in seeds:
            print(f"  [dim_scaling] Training seed={seed}, d={d}")
            config = _build_config_scaling(
                seed=seed,
                run_name=f"scaling_d{d}_D{hidden_dim}_G{granularity}_seed{seed}",
                train_steps=train_steps,
                visible_dim=d,
                hidden_dim=hidden_dim,
                basis_complexity=2,
                freq_sparsity=2,
                lattice_rl=True,
                lattice_granularity=granularity,
                max_horizon=horizon,
                device=device,
            )
            t0 = time.perf_counter()
            output = run_training(config, return_artifacts=True)
            wall_time = time.perf_counter() - t0
            print(f"  seed={seed} trained in {wall_time:.1f}s")

            model = output["model"]
            env = output["env"]
            random_rng = np.random.default_rng(int(seed) + 99_007)

            model.eval()
            for _task_i in range(num_tasks):
                spec = env.sample_episode_spec(
                    env_index=0, refresh_map=env.refresh_map_each_episode,
                )
                snapshot = _capture_spatial_task_snapshot(
                    env=env, source_xy=spec.source,
                    target_min_xy=spec.target_min_xy, env_index=0,
                )
                _apply_spatial_task_snapshot(env, snapshot, env_index=0)
                start_xy = snapshot["source_xy"].astype(np.float32)

                # RL hidden gradient
                rl_curve = _rollout_spatial_policy_curve(
                    model=model, env=env, device=device_t,
                    start_xy=start_xy, horizon=horizon, env_index=0,
                )
                method_curves["rl_hidden"].append(rl_curve)
                method_finals["rl_hidden"].append(float(rl_curve[-1]))

                # GD baseline (continuous space, tuned LR)
                gd_curve = _rollout_spatial_descent_curve(
                    env, start_xy, horizon, env_index=0,
                    base_lr=env.baseline_lr_gd,
                )
                method_curves["gd"].append(gd_curve)
                method_finals["gd"].append(float(gd_curve[-1]))

                # Adam baseline (continuous space, tuned LR)
                adam_curve = _rollout_spatial_adam_curve(
                    env, start_xy, horizon, env_index=0,
                    base_lr=env.baseline_lr_adam,
                )
                method_curves["adam"].append(adam_curve)
                method_finals["adam"].append(float(adam_curve[-1]))

                # Random search
                rs_curve = _rollout_random_search_curve_local(
                    env=env, start_xy=start_xy, horizon=horizon,
                    rng=random_rng, env_index=0,
                )
                method_curves["random_search"].append(rs_curve)
                method_finals["random_search"].append(float(rs_curve[-1]))

            rl_seed = np.mean(method_finals["rl_hidden"][-num_tasks:])
            gd_seed = np.mean(method_finals["gd"][-num_tasks:])
            adam_seed = np.mean(method_finals["adam"][-num_tasks:])
            rs_seed = np.mean(method_finals["random_search"][-num_tasks:])
            print(f"  seed={seed}: RL={rl_seed:.4f}, GD={gd_seed:.4f}, "
                  f"Adam={adam_seed:.4f}, RS={rs_seed:.4f}")
            del output

        for method in method_finals:
            arr = np.array(method_finals[method])
            stacked = np.stack(method_curves[method], axis=0)
            dim_results["methods"][method] = {
                **_stats(arr),
                "mean_curve": np.mean(stacked, axis=0).tolist(),
            }

        results["per_dim"][str(d)] = dim_results

    data_path = output_dir / "dimension_scaling_results.json"
    with data_path.open("w") as f:
        json.dump(results, f, indent=2)

    _plot_dimension_scaling(results, output_dir)
    print(f"[dim_scaling] Results saved to {output_dir}")
    return results


def _plot_dimension_scaling(results: dict, output_dir: Path) -> None:
    dims = results["visible_dims"]
    methods = ["rl_hidden", "gd", "adam", "random_search"]
    labels = {
        "rl_hidden": "RL hidden gradient",
        "gd": "GD (tuned LR)",
        "adam": "Adam (tuned LR)",
        "random_search": "Random search",
    }
    colors = {
        "rl_hidden": "#5B21B6",
        "gd": "#2563EB",
        "adam": "#059669",
        "random_search": "#D97706",
    }
    markers = {
        "rl_hidden": "o",
        "gd": "s",
        "adam": "D",
        "random_search": "^",
    }

    # --- Plot 1: Final objective vs dimension ---
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    _style_axis(ax)

    for method in methods:
        means = [results["per_dim"][str(d)]["methods"][method]["mean"] for d in dims]
        ci95s = [results["per_dim"][str(d)]["methods"][method]["ci95"] for d in dims]
        ax.errorbar(
            dims, means, yerr=ci95s,
            fmt=f"{markers[method]}-",
            color=colors[method],
            linewidth=2, markersize=7,
            capsize=4, capthick=1.5,
            label=labels[method],
        )

    ax.set_xlabel("Visible dimension $d$", fontsize=12)
    ax.set_ylabel("Final normalized objective", fontsize=12)
    ax.set_title("Dimension scaling ($K{=}2$, pairwise sparsity, lattice RL)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xticks(dims)
    fig.tight_layout()
    fig.savefig(output_dir / "dimension_scaling_final_obj.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Separation ratio vs dimension ---
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    _style_axis(ax)

    for method in ["gd", "adam", "random_search"]:
        ratios = []
        for d in dims:
            rl_mean = results["per_dim"][str(d)]["methods"]["rl_hidden"]["mean"]
            m_mean = results["per_dim"][str(d)]["methods"][method]["mean"]
            ratios.append(m_mean / max(rl_mean, 1e-6))
        ax.plot(
            dims, ratios,
            f"{markers[method]}-",
            color=colors[method],
            linewidth=2, markersize=7,
            label=labels[method],
        )

    ax.set_xlabel("Visible dimension $d$", fontsize=12)
    ax.set_ylabel("Ratio (baseline / RL hidden gradient)", fontsize=12)
    ax.set_title("Oracle separation ratio vs dimension", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(dims)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "dimension_scaling_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Per-dimension optimization curves ---
    n_dims = len(dims)
    ncols = min(3, n_dims)
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    for i, d in enumerate(dims):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        _style_axis(ax)
        for method in methods:
            curve = np.array(results["per_dim"][str(d)]["methods"][method]["mean_curve"])
            ax.plot(
                np.arange(len(curve)), curve,
                color=colors[method],
                linewidth=1.8,
                label=labels[method] if i == 0 else None,
            )
        hidden_dim = results["per_dim"][str(d)]["hidden_dim"]
        ax.set_title(f"$d={d}$, $D={hidden_dim}$", fontsize=11)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Objective", fontsize=9)

    for j in range(n_dims, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    handles, lbl = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbl, loc="lower center", ncol=len(methods), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Optimization curves by dimension", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "dimension_scaling_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness and generalization ablations for spatial task")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--train_steps", type=int, default=300_000)
    parser.add_argument("--num_tasks", type=int, default=200, help="Eval tasks per seed per condition")
    parser.add_argument("--max_horizon", type=int, default=60)
    parser.add_argument("--suite_output_dir", type=str, default="plots/spatial_robustness")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--studies", type=str, default="all",
        choices=["all", "oracle_noise", "hidden_dim", "unseen_maps", "dimension_scaling"],
    )
    parser.add_argument("--noise_levels", type=str, default="0.0,0.5,1.0,2.0,5.0,10.0")
    parser.add_argument("--basis_complexities", type=str, default="1,2,3,5")
    parser.add_argument("--visible_dims", type=str, default="2,3,4,5,6",
                        help="Comma-separated visible dims for dimension_scaling study")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]
    basis_complexities = [int(x.strip()) for x in args.basis_complexities.split(",") if x.strip()]
    visible_dims = [int(x.strip()) for x in args.visible_dims.split(",") if x.strip()]

    suite_root = Path(args.suite_output_dir).expanduser().resolve()
    suite_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "train_steps": args.train_steps,
        "num_tasks": args.num_tasks,
        "max_horizon": args.max_horizon,
        "studies": {},
    }

    run_all = args.studies == "all"

    if run_all or args.studies == "oracle_noise":
        noise_dir = suite_root / "oracle_noise"
        res = run_oracle_noise_study(
            seeds=seeds, train_steps=args.train_steps,
            num_tasks=args.num_tasks, horizon=args.max_horizon,
            noise_levels=noise_levels, output_dir=noise_dir,
            device=args.device,
        )
        manifest["studies"]["oracle_noise"] = {
            "dir": str(noise_dir),
            "noise_levels": noise_levels,
            "summary": {str(nl): res["per_noise_level"][str(nl)] for nl in noise_levels},
        }

    if run_all or args.studies == "hidden_dim":
        hdim_dir = suite_root / "hidden_dim"
        res = run_hidden_dim_study(
            seeds=seeds, train_steps=args.train_steps,
            num_tasks=args.num_tasks, horizon=args.max_horizon,
            basis_complexities=basis_complexities, output_dir=hdim_dir,
            device=args.device,
        )
        manifest["studies"]["hidden_dim"] = {
            "dir": str(hdim_dir),
            "basis_complexities": basis_complexities,
            "summary": {str(bc): res["per_basis_complexity"][str(bc)] for bc in basis_complexities},
        }

    if run_all or args.studies == "unseen_maps":
        unseen_dir = suite_root / "unseen_maps"
        res = run_unseen_maps_study(
            seeds=seeds, train_steps=args.train_steps,
            num_tasks=args.num_tasks, horizon=args.max_horizon,
            output_dir=unseen_dir, device=args.device,
        )
        manifest["studies"]["unseen_maps"] = {
            "dir": str(unseen_dir),
            "seen_map": res["seen_map"],
            "unseen_maps": res["unseen_maps"],
        }

    if run_all or args.studies == "dimension_scaling":
        scaling_dir = suite_root / "dimension_scaling"
        res = run_dimension_scaling_study(
            seeds=seeds, train_steps_base=args.train_steps,
            num_tasks=args.num_tasks, horizon=args.max_horizon,
            visible_dims=visible_dims, output_dir=scaling_dir,
            device=args.device,
        )
        manifest["studies"]["dimension_scaling"] = {
            "dir": str(scaling_dir),
            "visible_dims": visible_dims,
            "summary": {
                str(d): {
                    m: res["per_dim"][str(d)]["methods"][m]
                    for m in ["rl_hidden", "gd", "adam", "random_search"]
                }
                for d in visible_dims
            },
        }

    manifest_path = suite_root / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[robustness] All studies complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
