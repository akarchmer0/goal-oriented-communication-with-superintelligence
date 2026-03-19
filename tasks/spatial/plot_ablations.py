import argparse
import errno
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep matplotlib fully headless and in writable cache locations.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


MetricSpec = tuple[str, str, str, bool]

CURVE_METRICS: tuple[MetricSpec, ...] = (
    ("avg_final_objective", "objective", "Objective E(F(z))", False),
    ("avg_final_ref_distance", "distance", "Distance to reference minimum", False),
    ("success_rate", "success_rate", "Success rate", True),
)

CURVE_X_AXIS_MAX_EPISODES = 20_000.0


def _value_tag(value: int | float) -> str:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-12:
        return str(int(round(numeric)))
    return str(numeric).replace(".", "p").replace("-", "m")


def _value_label(ablation: str, value: int | float) -> str:
    if ablation in {"oracle_gradient_noise", "reward_noise"}:
        return f"sigma={float(value):g}"
    if ablation == "hidden_dim":
        return f"D={int(value)}"
    if ablation == "human_dim":
        return f"H={int(value)}"
    return f"value={value}"


def _read_jsonl(
    path: Path,
    *,
    timeout_retries: int = 3,
    initial_retry_sleep_sec: float = 0.35,
) -> list[dict[str, Any]]:
    timeout_errnos = {errno.ETIMEDOUT}
    if hasattr(errno, "WSAETIMEDOUT"):
        timeout_errnos.add(errno.WSAETIMEDOUT)

    for attempt in range(1, max(1, int(timeout_retries)) + 1):
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
        except OSError as exc:
            is_timeout = isinstance(exc, TimeoutError) or exc.errno in timeout_errnos
            if not is_timeout or attempt >= timeout_retries:
                raise
            sleep_sec = float(initial_retry_sleep_sec) * (2 ** (attempt - 1))
            print(
                f"[warn] Timed out while reading {path} (attempt {attempt}/{timeout_retries}); "
                f"retrying in {sleep_sec:.2f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_sec)

    return []


def _extract_metric_series(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    episodes: list[float] = []
    values: list[float] = []
    for row in rows:
        episode = row.get("episodes")
        value = row.get(key)
        if episode is None or value is None:
            continue
        try:
            ep_float = float(episode)
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ep_float) or not np.isfinite(value_float):
            continue
        episodes.append(ep_float)
        values.append(value_float)
    if not episodes:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    ep_arr = np.asarray(episodes, dtype=np.float64)
    val_arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(ep_arr, kind="mergesort")
    return ep_arr[order], val_arr[order]


def _align_series_to_grid(
    series: list[tuple[np.ndarray, np.ndarray]],
    n_grid: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [(x, y) for x, y in series if x.size > 0 and y.size > 0]
    if not valid:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )
    max_episode = max(float(np.max(x)) for x, _ in valid)
    grid = np.linspace(0.0, max_episode, num=max(20, int(n_grid)), dtype=np.float64)
    aligned: list[np.ndarray] = []
    for x, y in valid:
        aligned.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
    stacked = np.stack(aligned, axis=0)
    finite_mask = np.isfinite(stacked)
    finite_counts = np.sum(finite_mask, axis=0)
    summed = np.nansum(stacked, axis=0)
    mean_vals = np.divide(
        summed,
        finite_counts,
        out=np.full_like(summed, np.nan, dtype=np.float64),
        where=finite_counts > 0,
    )
    centered = np.where(finite_mask, stacked - mean_vals[None, :], 0.0)
    var = np.divide(
        np.sum(centered * centered, axis=0),
        finite_counts,
        out=np.full_like(summed, np.nan, dtype=np.float64),
        where=finite_counts > 0,
    )
    std_vals = np.sqrt(np.clip(var, 0.0, None))
    return grid, mean_vals, std_vals


def _make_axes(figsize: tuple[float, float] = (8.6, 5.0)) -> tuple[plt.Figure, plt.Axes]:
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


def _plot_curves_for_ablation(
    *,
    ablation_name: str,
    ablation_title: str,
    run_entries: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for entry in run_entries:
        grouped[float(entry["value"])].append(entry)
    sorted_values = sorted(grouped.keys())
    colors = plt.cm.viridis(np.linspace(0.10, 0.90, max(1, len(sorted_values))))

    curve_data: dict[str, Any] = {"ablation": ablation_name, "metrics": {}}
    plot_paths: list[str] = []
    metrics_rows_cache: dict[str, list[dict[str, Any]]] = {}
    unreadable_metrics_paths: set[str] = set()

    def _load_run_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
        metrics_path_str = str(run["metrics_jsonl"])
        if metrics_path_str in metrics_rows_cache:
            return metrics_rows_cache[metrics_path_str]
        if metrics_path_str in unreadable_metrics_paths:
            return []

        metrics_path = Path(metrics_path_str)
        try:
            rows = _read_jsonl(metrics_path)
        except (OSError, json.JSONDecodeError) as exc:
            unreadable_metrics_paths.add(metrics_path_str)
            print(
                "[warn] Skipping run due to unreadable metrics file: "
                f"ablation={ablation_name} value={run['value']} seed={run['seed']} "
                f"path={metrics_path} error={exc}",
                file=sys.stderr,
            )
            return []

        metrics_rows_cache[metrics_path_str] = rows
        return rows

    for metric_key, metric_slug, y_label, is_percent in CURVE_METRICS:
        metric_data: dict[str, Any] = {}

        fig, ax = _make_axes()
        for value, color in zip(sorted_values, colors):
            label = _value_label(ablation_name, value)
            seed_runs = sorted(grouped[value], key=lambda x: int(x["seed"]))
            for run in seed_runs:
                rows = _load_run_rows(run)
                ep, vals = _extract_metric_series(rows, metric_key)
                if ep.size == 0:
                    continue
                ax.plot(ep, vals, color=color, linewidth=1.0, alpha=0.24)
            ax.plot([], [], color=color, linewidth=2.1, alpha=0.95, label=label)

        ax.set_title(f"{ablation_title} | {y_label} (individual seeds)", loc="left", fontsize=11, pad=10)
        ax.set_xlabel("Training episodes")
        ax.set_ylabel(y_label)
        ax.set_xlim(0.0, CURVE_X_AXIS_MAX_EPISODES)
        if is_percent:
            ax.set_ylim(0.0, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout()
        individual_path = output_dir / f"{metric_slug}_individual.png"
        fig.savefig(individual_path, dpi=190)
        plt.close(fig)
        plot_paths.append(str(individual_path.resolve()))

        fig, ax = _make_axes()
        for value, color in zip(sorted_values, colors):
            label = _value_label(ablation_name, value)
            seed_runs = sorted(grouped[value], key=lambda x: int(x["seed"]))
            series: list[tuple[np.ndarray, np.ndarray]] = []
            run_data: list[dict[str, Any]] = []
            for run in seed_runs:
                rows = _load_run_rows(run)
                ep, vals = _extract_metric_series(rows, metric_key)
                if ep.size == 0:
                    continue
                series.append((ep, vals))
                run_data.append(
                    {
                        "seed": int(run["seed"]),
                        "episodes": ep.tolist(),
                        "values": vals.tolist(),
                    }
                )
            grid, mean_vals, std_vals = _align_series_to_grid(series)
            if grid.size == 0:
                continue
            ax.plot(grid, mean_vals, color=color, linewidth=2.0, label=label)
            ax.fill_between(
                grid,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.20,
                linewidth=0,
            )
            metric_data[str(value)] = {
                "label": label,
                "mean_grid_episodes": grid.tolist(),
                "mean_values": mean_vals.tolist(),
                "std_values": std_vals.tolist(),
                "runs": run_data,
            }

        ax.set_title(f"{ablation_title} | {y_label} (mean +/- std)", loc="left", fontsize=11, pad=10)
        ax.set_xlabel("Training episodes")
        ax.set_ylabel(y_label)
        ax.set_xlim(0.0, CURVE_X_AXIS_MAX_EPISODES)
        if is_percent:
            ax.set_ylim(0.0, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout()
        mean_path = output_dir / f"{metric_slug}_mean.png"
        fig.savefig(mean_path, dpi=190)
        plt.close(fig)
        plot_paths.append(str(mean_path.resolve()))

        curve_data["metrics"][metric_slug] = metric_data

    return {"plot_paths": plot_paths, "curve_data": curve_data}


def _plot_heatmaps_for_ablation(
    *,
    ablation_name: str,
    ablation_title: str,
    run_entries: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for entry in run_entries:
        heatmap_json = entry.get("heatmap_json")
        if heatmap_json:
            grouped[float(entry["value"])].append(entry)

    if not grouped:
        return {"plot_paths": [], "heatmap_data": {"ablation": ablation_name, "values": {}}}

    sorted_values = sorted(grouped.keys())
    all_plot_paths: list[str] = []
    heatmap_data: dict[str, Any] = {"ablation": ablation_name, "values": {}}

    for value in sorted_values:
        runs = sorted(grouped[value], key=lambda x: int(x["seed"]))
        loaded: list[tuple[int, dict[str, Any]]] = []
        for run in runs:
            heatmap_path = Path(str(run["heatmap_json"]))
            try:
                data = json.loads(heatmap_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(
                    "[warn] Skipping unreadable heatmap file: "
                    f"ablation={ablation_name} value={run['value']} seed={run['seed']} "
                    f"path={heatmap_path} error={exc}",
                    file=sys.stderr,
                )
                continue
            loaded.append((int(run["seed"]), data))
        if not loaded:
            continue

        norm_arrays = [np.asarray(data["normalized_counts"], dtype=np.float64) for _, data in loaded]
        coord_limit = float(loaded[0][1]["coord_limit"])
        extent = [-coord_limit, coord_limit, -coord_limit, coord_limit]
        mean_norm = np.mean(np.stack(norm_arrays, axis=0), axis=0)

        cols = min(4, len(loaded))
        rows = int(math.ceil(len(loaded) / max(1, cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.5 * rows), squeeze=False)
        for idx, (seed, data) in enumerate(loaded):
            ax = axes[idx // cols][idx % cols]
            norm_counts = np.asarray(data["normalized_counts"], dtype=np.float64)
            im = ax.imshow(
                norm_counts,
                origin="lower",
                extent=extent,
                cmap="magma",
                aspect="equal",
            )
            ax.set_title(f"seed {seed}", fontsize=9)
            ax.set_xlabel("z[0]")
            ax.set_ylabel("z[1]")
        for idx in range(len(loaded), rows * cols):
            axes[idx // cols][idx % cols].axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Normalized occupancy")
        fig.suptitle(
            f"{ablation_title} | {_value_label(ablation_name, value)} | trajectory heatmaps (individual)",
            fontsize=11,
            x=0.02,
            ha="left",
        )
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.90, wspace=0.28, hspace=0.35)
        individual_path = output_dir / f"heatmap_{_value_tag(value)}_individual.png"
        fig.savefig(individual_path, dpi=190)
        plt.close(fig)
        all_plot_paths.append(str(individual_path.resolve()))

        fig, ax = plt.subplots(figsize=(6.3, 5.6))
        im = ax.imshow(
            mean_norm,
            origin="lower",
            extent=extent,
            cmap="magma",
            aspect="equal",
        )
        ax.set_title(
            f"{ablation_title} | {_value_label(ablation_name, value)} | trajectory heatmap (mean)",
            loc="left",
            fontsize=11,
            pad=10,
        )
        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Normalized occupancy")
        fig.tight_layout()
        mean_path = output_dir / f"heatmap_{_value_tag(value)}_mean.png"
        fig.savefig(mean_path, dpi=190)
        plt.close(fig)
        all_plot_paths.append(str(mean_path.resolve()))

        heatmap_data["values"][str(value)] = {
            "label": _value_label(ablation_name, value),
            "coord_limit": coord_limit,
            "mean_normalized_counts": mean_norm.tolist(),
            "seeds": [seed for seed, _ in loaded],
        }

    return {"plot_paths": all_plot_paths, "heatmap_data": heatmap_data}


def _resolve_output_path(raw: str | None, *, manifest_dir: Path, fallback: Path) -> Path:
    if raw is None or str(raw).strip() == "":
        return fallback.resolve()
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_dir / candidate).resolve()


def plot_from_manifest(
    *,
    manifest_path: Path,
    plots_root_override: str | None = None,
    plot_data_root_override: str | None = None,
    write_manifest: bool = True,
    write_summary: bool = True,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    suite_root = _resolve_output_path(
        manifest.get("suite_root"),
        manifest_dir=manifest_dir,
        fallback=manifest_dir,
    )
    plots_root = _resolve_output_path(
        plots_root_override if plots_root_override is not None else manifest.get("plots_root"),
        manifest_dir=manifest_dir,
        fallback=suite_root / "plots",
    )
    plot_data_root = _resolve_output_path(
        plot_data_root_override if plot_data_root_override is not None else manifest.get("plot_data_root"),
        manifest_dir=manifest_dir,
        fallback=suite_root / "plot_data",
    )
    plots_root.mkdir(parents=True, exist_ok=True)
    plot_data_root.mkdir(parents=True, exist_ok=True)

    manifest["plots_root"] = str(plots_root.resolve())
    manifest["plot_data_root"] = str(plot_data_root.resolve())
    manifest["last_plotted_at_utc"] = datetime.now(timezone.utc).isoformat()

    ablations = manifest.get("ablations", {})
    for ablation_name, ablation_meta in ablations.items():
        if not isinstance(ablation_meta, dict):
            continue
        ablation_title = str(ablation_meta.get("title", ablation_name))
        run_entries = list(ablation_meta.get("runs", []))
        include_heatmaps = bool(ablation_meta.get("include_heatmaps", False))

        ablation_plot_dir = plots_root / str(ablation_name)
        curve_result = _plot_curves_for_ablation(
            ablation_name=str(ablation_name),
            ablation_title=ablation_title,
            run_entries=run_entries,
            output_dir=ablation_plot_dir / "curves",
        )
        curve_data_path = plot_data_root / f"{ablation_name}_curves.json"
        with curve_data_path.open("w", encoding="utf-8") as handle:
            json.dump(curve_result["curve_data"], handle, indent=2)

        heatmap_result = {"plot_paths": [], "heatmap_data": {"ablation": ablation_name, "values": {}}}
        heatmap_data_path = None
        if include_heatmaps:
            heatmap_result = _plot_heatmaps_for_ablation(
                ablation_name=str(ablation_name),
                ablation_title=ablation_title,
                run_entries=run_entries,
                output_dir=ablation_plot_dir / "heatmaps",
            )
            heatmap_data_path = plot_data_root / f"{ablation_name}_heatmaps.json"
            with heatmap_data_path.open("w", encoding="utf-8") as handle:
                json.dump(heatmap_result["heatmap_data"], handle, indent=2)

        ablation_meta["curve_plot_paths"] = curve_result["plot_paths"]
        ablation_meta["heatmap_plot_paths"] = heatmap_result["plot_paths"]
        ablation_meta["curve_plot_data_json"] = str(curve_data_path.resolve())
        ablation_meta["heatmap_plot_data_json"] = (
            str(heatmap_data_path.resolve()) if heatmap_data_path else None
        )

    if write_manifest:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    suite_summary = {
        "manifest_path": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "plots_root": str(plots_root.resolve()),
        "plot_data_root": str(plot_data_root.resolve()),
        "num_seeds": len(manifest.get("seeds", [])),
        "seeds": [int(seed) for seed in manifest.get("seeds", [])],
        "ablations": list(ablations.keys()),
        "plot_cmd": f"python3 -m tasks.spatial.plot_ablations --manifest {manifest_path}",
    }
    if write_summary:
        summary_path = suite_root / "suite_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(suite_summary, handle, indent=2)
        suite_summary["summary_path"] = str(summary_path.resolve())

    return suite_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or regenerate spatial ablation plots from a suite manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to suite manifest.json created by tasks.spatial.run_ablation_suite.",
    )
    parser.add_argument(
        "--plots_root",
        type=str,
        default=None,
        help="Optional override for plot image output root.",
    )
    parser.add_argument(
        "--plot_data_root",
        type=str,
        default=None,
        help="Optional override for plot-data JSON output root.",
    )
    parser.add_argument(
        "--no_write_manifest",
        action="store_true",
        help="Do not write updated plot paths back into the manifest.",
    )
    parser.add_argument(
        "--no_write_summary",
        action="store_true",
        help="Do not write suite_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = plot_from_manifest(
        manifest_path=Path(args.manifest),
        plots_root_override=args.plots_root,
        plot_data_root_override=args.plot_data_root,
        write_manifest=not bool(args.no_write_manifest),
        write_summary=not bool(args.no_write_summary),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
