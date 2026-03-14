import argparse
import json
import math
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep matplotlib fully headless and in writable cache locations.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

from v2.config import TrainConfig
from v2.oracle import SpatialOracle
from v2.spatial_env import VectorizedSpatialEnv
from v2.train import _resolve_device, run_training


def _parse_int_list(raw: str, arg_name: str, min_value: int | None = None) -> list[int]:
    values: list[int] = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{arg_name} contains non-integer token: {token!r}") from exc
        if min_value is not None and value < min_value:
            raise ValueError(f"{arg_name} must be >= {min_value}, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must include at least one integer")
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _parse_float_list(raw: str, arg_name: str, min_value: float | None = None) -> list[float]:
    values: list[float] = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"{arg_name} contains non-float token: {token!r}") from exc
        if min_value is not None and value < min_value:
            raise ValueError(f"{arg_name} must be >= {min_value}, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must include at least one float")
    deduped: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = float(value)
        if key in seen:
            continue
        deduped.append(key)
        seen.add(key)
    return deduped


def _value_tag(value: int | float) -> str:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-12:
        return str(int(round(numeric)))
    return str(numeric).replace(".", "p").replace("-", "m")


def _build_eval_env(config: TrainConfig) -> VectorizedSpatialEnv:
    spatial_token_dim = config.spatial_token_dim
    if config.oracle_mode == "convex_gradient":
        spatial_token_dim = config.spatial_hidden_dim
    elif config.oracle_mode == "visible_gradient":
        spatial_token_dim = config.spatial_visible_dim
    oracle = SpatialOracle(
        hidden_dim=config.spatial_hidden_dim,
        token_dim=spatial_token_dim,
        mode=config.oracle_mode,
        seed=config.seed + 10_029,
        token_noise_std=config.spatial_token_noise_std,
    )
    return VectorizedSpatialEnv(
        hidden_dim=config.spatial_hidden_dim,
        visible_dim=config.spatial_visible_dim,
        coord_limit=config.spatial_coord_limit,
        oracle=oracle,
        n_env=1,
        sensing=config.sensing,
        max_horizon=config.max_horizon,
        seed=config.seed + 10_041,
        s1_step_penalty=config.s1_step_penalty,
        reward_noise_std=config.reward_noise_std,
        step_size=config.spatial_step_size,
        success_threshold=config.spatial_success_threshold,
        basis_complexity=config.spatial_basis_complexity,
        f_type=config.spatial_f_type,
        refresh_map_each_episode=config.spatial_refresh_map_each_episode,
        compute_episode_baselines=False,
    )


@torch.no_grad()
def _rollout_policy_episode(
    model: torch.nn.Module,
    env: VectorizedSpatialEnv,
    device: torch.device,
) -> tuple[np.ndarray, bool, float]:
    spec = env.sample_episode_spec(
        env_index=0,
        refresh_map=env.refresh_map_each_episode,
    )
    state = spec.source.copy()
    hidden_state = model.initial_state(batch_size=1, device=device)
    trajectory = [state.astype(np.float32)]

    for step in range(int(spec.horizon)):
        token_features = env._obs_token_features(state, env_index=0)[None, :]
        dist_feature = env._normalized_objective_value(state, env_index=0)
        step_fraction = float(step / max(1, int(spec.horizon)))

        token_t = torch.tensor(token_features, dtype=torch.float32, device=device)
        dist_t = torch.tensor([dist_feature], dtype=torch.float32, device=device)
        step_t = torch.tensor([step_fraction], dtype=torch.float32, device=device)
        action_t, _, _, hidden_state = model.act(
            token_t,
            dist_t,
            step_t,
            hidden_state=hidden_state,
            deterministic=True,
        )
        action = action_t.squeeze(0).cpu().numpy()
        state = env._apply_action(state, action)
        trajectory.append(state.astype(np.float32))
        if env._is_success(state, env_index=0):
            break

    final_objective_raw = float(env._objective_value(state, env_index=0))
    final_objective = env._normalized_objective_from_raw(final_objective_raw, env_index=0)
    success = bool(final_objective <= env.success_threshold)
    return np.asarray(trajectory, dtype=np.float32), success, final_objective


def _collect_trajectory_rollouts(
    model: torch.nn.Module,
    config: TrainConfig,
    output_jsonl: Path,
    output_heatmap_json: Path,
    num_episodes: int,
    heatmap_bins: int,
) -> dict[str, Any]:
    if config.spatial_visible_dim != 2:
        return {
            "trajectory_jsonl": None,
            "heatmap_json": None,
            "num_eval_episodes": 0,
            "success_rate_eval": None,
        }

    env = _build_eval_env(config)
    device = _resolve_device(config.device)
    was_training = bool(model.training)
    model.eval()

    all_points: list[np.ndarray] = []
    successes = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for episode in range(int(num_episodes)):
            trajectory, success, final_objective = _rollout_policy_episode(model, env, device)
            if success:
                successes += 1
            all_points.append(trajectory[:, :2])
            handle.write(
                json.dumps(
                    {
                        "episode": int(episode),
                        "success": bool(success),
                        "final_objective": float(final_objective),
                        "trajectory": trajectory.tolist(),
                    }
                )
                + "\n"
            )

    if was_training:
        model.train()

    stacked_points = (
        np.concatenate(all_points, axis=0)
        if all_points
        else np.zeros((0, 2), dtype=np.float32)
    )
    coord_limit = float(config.spatial_coord_limit)
    bins = max(16, int(heatmap_bins))
    hist, x_edges, y_edges = np.histogram2d(
        stacked_points[:, 0] if stacked_points.size else np.asarray([], dtype=np.float32),
        stacked_points[:, 1] if stacked_points.size else np.asarray([], dtype=np.float32),
        bins=bins,
        range=[[-coord_limit, coord_limit], [-coord_limit, coord_limit]],
    )
    counts = hist.T.astype(np.float64)
    total = float(np.sum(counts))
    normalized = counts / total if total > 0.0 else counts

    output_heatmap_json.parent.mkdir(parents=True, exist_ok=True)
    with output_heatmap_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "coord_limit": coord_limit,
                "bins": bins,
                "num_points": int(stacked_points.shape[0]),
                "x_edges": x_edges.tolist(),
                "y_edges": y_edges.tolist(),
                "counts": counts.tolist(),
                "normalized_counts": normalized.tolist(),
            },
            handle,
            indent=2,
        )

    return {
        "trajectory_jsonl": str(output_jsonl.resolve()),
        "heatmap_json": str(output_heatmap_json.resolve()),
        "num_eval_episodes": int(num_episodes),
        "success_rate_eval": float(successes / max(1, int(num_episodes))),
    }


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run spatial-only PPO-oracle ablation suite and record JSON/JSONL artifacts. "
            "Plots can be generated or regenerated with v2.plot_ablations."
        )
    )
    parser.add_argument(
        "--seeds",
        type=str,
        required=True,
        help="Comma-separated random seeds to run for each ablation value.",
    )
    parser.add_argument(
        "--suite_output_dir",
        type=str,
        default="plots/spatial_ablation_suite",
        help="Directory for suite manifest, plot data JSON, and generated plots.",
    )
    parser.add_argument(
        "--suite_name",
        type=str,
        default="spatial_ablation_suite",
        help="Name tag stored in manifest and used in run names.",
    )
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.device,
        help="Device for training (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument("--run_name_prefix", type=str, default="spatial_ablation")

    parser.add_argument("--train_steps", type=int, default=defaults.train_steps)
    parser.add_argument("--n_env", type=int, default=defaults.n_env)
    parser.add_argument("--rollout_len", type=int, default=defaults.rollout_len)
    parser.add_argument("--running_avg_window", type=int, default=defaults.running_avg_window)
    parser.add_argument(
        "--save_metrics_interval_episodes",
        type=int,
        default=defaults.save_metrics_interval_episodes,
    )
    parser.add_argument("--eval_interval_episodes", type=int, default=defaults.eval_interval_episodes)
    parser.add_argument("--max_horizon", type=int, default=defaults.max_horizon)

    parser.add_argument("--sensing", type=str, choices=["S0", "S1"], default=defaults.sensing)
    parser.add_argument(
        "--oracle_mode",
        type=str,
        choices=[
            "convex_gradient",
            "linear_embedding",
            "fresh_linear_embedding",
            "random_message",
        ],
        default="convex_gradient",
        help="Spatial oracle mode used for all ablations.",
    )

    parser.add_argument("--policy_hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument("--spatial_coord_limit", type=int, default=defaults.spatial_coord_limit)
    parser.add_argument(
        "--base_lr",
        dest="spatial_step_size",
        type=float,
        default=defaults.spatial_step_size,
        help="Base learning rate for visible-space GD/Adam baselines.",
    )
    parser.add_argument(
        "--spatial_step_size",
        dest="spatial_step_size",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--spatial_success_threshold",
        type=float,
        default=defaults.spatial_success_threshold,
    )
    parser.add_argument(
        "--spatial_basis_complexity",
        type=int,
        default=defaults.spatial_basis_complexity,
    )
    parser.add_argument(
        "--spatial_f_type",
        type=str,
        choices=["FOURIER", "MLP"],
        default=defaults.spatial_f_type,
    )
    parser.add_argument(
        "--spatial_policy_arch",
        type=str,
        choices=["mlp", "gru"],
        default=defaults.spatial_policy_arch,
    )
    parser.add_argument(
        "--spatial_refresh_map_each_episode",
        action="store_true",
        default=defaults.spatial_refresh_map_each_episode,
    )

    parser.add_argument(
        "--oracle_noise_sigmas",
        type=str,
        default="0,1,5,10,50,100",
        help="Comma-separated sigmas for oracle gradient-token noise sweep.",
    )
    parser.add_argument(
        "--reward_noise_sigmas",
        type=str,
        default="0,1,5,10,50,100",
        help="Comma-separated sigmas for sensing/reward noise sweep.",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="10,100,1000",
        help="Comma-separated hidden-space dimensions D.",
    )
    parser.add_argument(
        "--human_dims",
        type=str,
        default="2,4,8,16",
        help="Comma-separated human/control-space dimensions H.",
    )
    parser.add_argument(
        "--h_sweep_hidden_dim",
        type=int,
        default=100,
        help="Fixed hidden-space D used for the H sweep.",
    )
    parser.add_argument(
        "--h_sweep_budget_scaling",
        type=str,
        choices=["none", "step_size", "max_horizon", "both"],
        default="step_size",
        help=(
            "For H sweep only, scale movement budget by sqrt(H / h_sweep_reference_dim). "
            "'step_size' scales step size, 'max_horizon' scales horizon, 'both' scales both."
        ),
    )
    parser.add_argument(
        "--h_sweep_reference_dim",
        type=float,
        default=2.0,
        help="Reference H for H-sweep budget scaling factor sqrt(H / reference_dim).",
    )
    parser.add_argument(
        "--default_hidden_dim",
        type=int,
        default=100,
        help="Fixed hidden-space D used for non-D sweeps.",
    )
    parser.add_argument(
        "--default_visible_dim",
        type=int,
        default=2,
        help="Visible/control-space dimension H used for non-H sweeps.",
    )
    parser.add_argument(
        "--trajectory_eval_episodes",
        type=int,
        default=200,
        help="Number of post-training rollout episodes used to build heatmaps.",
    )
    parser.add_argument(
        "--trajectory_heatmap_bins",
        type=int,
        default=100,
        help="Number of bins per axis for trajectory heatmaps.",
    )
    parser.add_argument(
        "--skip_plotting",
        action="store_true",
        help="Skip automatic plotting; use v2.plot_ablations on the manifest later.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds, "seeds")
    oracle_noise_sigmas = _parse_float_list(args.oracle_noise_sigmas, "oracle_noise_sigmas", min_value=0.0)
    reward_noise_sigmas = _parse_float_list(args.reward_noise_sigmas, "reward_noise_sigmas", min_value=0.0)
    hidden_dims = _parse_int_list(args.hidden_dims, "hidden_dims", min_value=1)
    human_dims = _parse_int_list(args.human_dims, "human_dims", min_value=2)
    if args.default_visible_dim < 2:
        raise ValueError("default_visible_dim must be >= 2")
    if args.h_sweep_hidden_dim < 1:
        raise ValueError("h_sweep_hidden_dim must be >= 1")
    if args.default_hidden_dim < 1:
        raise ValueError("default_hidden_dim must be >= 1")
    if args.h_sweep_reference_dim <= 0.0:
        raise ValueError("h_sweep_reference_dim must be > 0")

    suite_root = Path(args.suite_output_dir).expanduser().resolve()
    plots_root = suite_root / "plots"
    plot_data_root = suite_root / "plot_data"
    suite_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    plot_data_root.mkdir(parents=True, exist_ok=True)

    base_config = TrainConfig(
        task="spatial",
        train_steps=int(args.train_steps),
        n_env=int(args.n_env),
        rollout_len=int(args.rollout_len),
        running_avg_window=int(args.running_avg_window),
        save_metrics_interval_episodes=int(args.save_metrics_interval_episodes),
        eval_interval_episodes=int(args.eval_interval_episodes),
        max_horizon=int(args.max_horizon),
        logdir=str(args.logdir),
        sensing=str(args.sensing),
        oracle_mode=str(args.oracle_mode),
        lr=float(args.lr),
        ppo_epochs=int(args.ppo_epochs),
        minibatches=int(args.minibatches),
        hidden_dim=int(args.policy_hidden_dim),
        spatial_coord_limit=int(args.spatial_coord_limit),
        spatial_step_size=float(args.spatial_step_size),
        spatial_success_threshold=float(args.spatial_success_threshold),
        spatial_basis_complexity=int(args.spatial_basis_complexity),
        spatial_f_type=str(args.spatial_f_type),
        spatial_policy_arch=str(args.spatial_policy_arch),
        spatial_refresh_map_each_episode=bool(args.spatial_refresh_map_each_episode),
        spatial_plot_interval_episodes=0,
        spatial_enable_baselines=False,
        enable_training_plots=False,
        spatial_hidden_dim=int(args.default_hidden_dim),
        spatial_visible_dim=int(args.default_visible_dim),
        spatial_token_dim=int(args.default_hidden_dim),
        device=str(args.device),
    )

    ablations: list[dict[str, Any]] = [
        {
            "name": "oracle_gradient_noise",
            "title": "Oracle gradient-noise sweep",
            "parameter": "spatial_token_noise_std",
            "values": oracle_noise_sigmas,
            "include_heatmaps": True,
        },
        {
            "name": "reward_noise",
            "title": "Sensing/reward-noise sweep",
            "parameter": "reward_noise_std",
            "values": reward_noise_sigmas,
            "include_heatmaps": True,
        },
        {
            "name": "hidden_dim",
            "title": "Hidden-dimension sweep",
            "parameter": "spatial_hidden_dim",
            "values": hidden_dims,
            "include_heatmaps": True,
        },
        # {
        #     "name": "human_dim",
        #     "title": "Human/control-space dimension sweep",
        #     "parameter": "spatial_visible_dim",
        #     "values": human_dims,
        #     "include_heatmaps": False,
        # },
    ]

    manifest: dict[str, Any] = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_name": str(args.suite_name),
        "suite_root": str(suite_root),
        "plots_root": str(plots_root),
        "plot_data_root": str(plot_data_root),
        "seeds": [int(seed) for seed in seeds],
        "base_config": asdict(base_config),
        "trajectory_eval_episodes": int(args.trajectory_eval_episodes),
        "trajectory_heatmap_bins": int(args.trajectory_heatmap_bins),
        "h_sweep_budget_scaling": str(args.h_sweep_budget_scaling),
        "h_sweep_reference_dim": float(args.h_sweep_reference_dim),
        "ablations": {},
    }

    for ablation in ablations:
        ablation_name = str(ablation["name"])
        ablation_title = str(ablation["title"])
        values = list(ablation["values"])
        include_heatmaps = bool(ablation["include_heatmaps"])
        run_entries: list[dict[str, Any]] = []

        for value in values:
            for seed in seeds:
                run_name = (
                    f"{args.run_name_prefix}_{ablation_name}_{_value_tag(value)}_seed{int(seed)}"
                )

                h_sweep_scale = 1.0
                if ablation_name == "human_dim":
                    h_sweep_scale = math.sqrt(float(value) / float(args.h_sweep_reference_dim))
                scaled_step_size = float(args.spatial_step_size)
                scaled_max_horizon = int(args.max_horizon)
                if ablation_name == "human_dim" and args.h_sweep_budget_scaling in {"step_size", "both"}:
                    scaled_step_size = float(args.spatial_step_size) * h_sweep_scale
                if ablation_name == "human_dim" and args.h_sweep_budget_scaling in {"max_horizon", "both"}:
                    scaled_max_horizon = max(1, int(round(float(args.max_horizon) * h_sweep_scale)))

                config = replace(
                    base_config,
                    seed=int(seed),
                    run_name=run_name,
                    max_horizon=scaled_max_horizon,
                    spatial_step_size=scaled_step_size,
                    spatial_hidden_dim=(
                        int(value)
                        if ablation_name == "hidden_dim"
                        else int(args.h_sweep_hidden_dim)
                        if ablation_name == "human_dim"
                        else int(args.default_hidden_dim)
                    ),
                    spatial_visible_dim=(
                        int(value)
                        if ablation_name == "human_dim"
                        else int(args.default_visible_dim)
                    ),
                    spatial_token_noise_std=(
                        float(value) if ablation_name == "oracle_gradient_noise" else 0.0
                    ),
                    reward_noise_std=(
                        float(value) if ablation_name == "reward_noise" else 0.0
                    ),
                    spatial_token_dim=(
                        int(value)
                        if ablation_name == "hidden_dim"
                        else int(args.h_sweep_hidden_dim)
                        if ablation_name == "human_dim"
                        else int(args.default_hidden_dim)
                    ),
                )

                need_model = include_heatmaps and config.spatial_visible_dim == 2
                output = run_training(config, return_artifacts=need_model)
                run_dir = Path(output["summary"]["run_dir"]).expanduser().resolve()

                run_entry: dict[str, Any] = {
                    "ablation": ablation_name,
                    "ablation_title": ablation_title,
                    "parameter": str(ablation["parameter"]),
                    "value": float(value),
                    "seed": int(seed),
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "config_json": str((run_dir / "config.json").resolve()),
                    "metrics_jsonl": str((run_dir / "metrics.jsonl").resolve()),
                    "summary_json": str((run_dir / "summary.json").resolve()),
                    "trajectory_jsonl": None,
                    "heatmap_json": None,
                    "h_sweep_budget_scale_factor": (
                        float(h_sweep_scale) if ablation_name == "human_dim" else None
                    ),
                    "effective_max_horizon": int(config.max_horizon),
                    "effective_base_lr": float(config.spatial_step_size),
                    # Backward-compatible key retained for existing downstream tooling.
                    "effective_step_size": float(
                        output["summary"].get("spatial_effective_step_size", config.spatial_step_size)
                    ),
                }

                if need_model:
                    trajectory_jsonl = run_dir / "oracle_eval_trajectories.jsonl"
                    heatmap_json = run_dir / "oracle_eval_heatmap.json"
                    trajectory_summary = _collect_trajectory_rollouts(
                        model=output["model"],
                        config=config,
                        output_jsonl=trajectory_jsonl,
                        output_heatmap_json=heatmap_json,
                        num_episodes=int(args.trajectory_eval_episodes),
                        heatmap_bins=int(args.trajectory_heatmap_bins),
                    )
                    run_entry["trajectory_jsonl"] = trajectory_summary["trajectory_jsonl"]
                    run_entry["heatmap_json"] = trajectory_summary["heatmap_json"]
                    run_entry["trajectory_eval"] = {
                        "num_eval_episodes": int(trajectory_summary["num_eval_episodes"]),
                        "success_rate_eval": trajectory_summary["success_rate_eval"],
                    }
                    del output["model"]

                run_entries.append(run_entry)

        manifest["ablations"][ablation_name] = {
            "title": ablation_title,
            "parameter": str(ablation["parameter"]),
            "values": [float(v) for v in values],
            "include_heatmaps": include_heatmaps,
            "runs": run_entries,
            "curve_plot_paths": [],
            "heatmap_plot_paths": [],
            "curve_plot_data_json": None,
            "heatmap_plot_data_json": None,
        }

    manifest_path = suite_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if args.skip_plotting:
        suite_summary = {
            "manifest_path": str(manifest_path.resolve()),
            "suite_root": str(suite_root),
            "num_seeds": len(seeds),
            "seeds": [int(seed) for seed in seeds],
            "ablations": list(manifest["ablations"].keys()),
            "plotting_skipped": True,
            "plot_cmd": f"python3 -m v2.plot_ablations --manifest {manifest_path}",
        }
        summary_path = suite_root / "suite_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(suite_summary, handle, indent=2)
    else:
        from v2.plot_ablations import plot_from_manifest

        suite_summary = plot_from_manifest(manifest_path=manifest_path)

    print(json.dumps(suite_summary, indent=2))


if __name__ == "__main__":
    main()
