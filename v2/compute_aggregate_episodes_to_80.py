#!/usr/bin/env python3
"""
Compute the episode where the aggregate learning curve (mean across seeds) crosses 0.8.

This matches what the plot visually shows, unlike the JSON's episodes_to_80_reached_only
which is the mean of per-seed crossing points.

Usage:
    python -m v2.compute_aggregate_episodes_to_80 --manifest plots/requested_experiments_manifest_S0_n10000_w500_seeds30.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_metrics_fast(run_dir: Path, step: int = 50) -> list[dict]:
    """Load metrics from CSV (preferred) or JSONL with downsampling for speed."""
    csv_path = run_dir / "metrics.csv"
    jsonl_path = run_dir / "metrics.jsonl"
    rows: list[dict] = []
    last_row: dict | None = None

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    last_row = {
                        "episodes": int(row["episodes"]),
                        "success_rate": float(row["success_rate"]),
                    }
                    if i % step == 0:
                        rows.append(last_row)
                except (KeyError, ValueError):
                    continue
    elif jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if line.strip():
                    last_row = json.loads(line.strip())
                    if i % step == 0:
                        rows.append({
                            "episodes": int(last_row["episodes"]),
                            "success_rate": float(last_row["success_rate"]),
                        })
    else:
        return []

    if step > 1 and last_row and rows and last_row["episodes"] != rows[-1]["episodes"]:
        rows.append(last_row)
    return rows


def _extract_curve(metrics: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not metrics:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    episodes = np.asarray([float(item["episodes"]) for item in metrics], dtype=np.float64)
    success_rate = np.asarray(
        [float(item["success_rate"]) for item in metrics], dtype=np.float64
    )
    mask = np.isfinite(episodes) & np.isfinite(success_rate)
    episodes = episodes[mask]
    success_rate = np.clip(success_rate[mask], 0.0, 1.0)
    return episodes, success_rate


def _aggregate_curve(
    seed_metrics: list[list[dict]], max_points: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute aggregate curve (mean success rate across seeds at each episode)."""
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for metrics in seed_metrics:
        episodes, success_rate = _extract_curve(metrics)
        if episodes.size == 0:
            continue
        curves.append((episodes, success_rate))

    if not curves:
        return None

    max_common_episode = int(min(float(episodes[-1]) for episodes, _ in curves))
    if max_common_episode < 1:
        return None

    num_points = int(min(max_points, max_common_episode))
    num_points = max(1, num_points)
    x_grid = np.linspace(
        1.0, float(max_common_episode), num=num_points, dtype=np.float64
    )

    interpolated = np.vstack(
        [
            np.interp(x_grid, episodes, success_rate)
            for episodes, success_rate in curves
        ]
    )
    mean_curve = interpolated.mean(axis=0)
    return x_grid, mean_curve


def _find_aggregate_crossing(
    x_grid: np.ndarray, mean_curve: np.ndarray, threshold: float = 0.8
) -> int | None:
    """First episode index where mean_curve >= threshold."""
    above = mean_curve >= threshold
    if not np.any(above):
        return None
    first_idx = int(np.argmax(above))
    return int(round(float(x_grid[first_idx])))


def _load_seed_metrics(
    seed_records: list[dict], downsample_step: int = 50
) -> list[list[dict]]:
    """Load metrics, downsampling for speed (step=50 gives ~1700 pts per 85k-ep run)."""
    metrics_by_seed: list[list[dict]] = []
    for item in seed_records:
        run_dir = _resolve_path(item["run_dir"])
        if not (run_dir / "metrics.csv").exists() and not (run_dir / "metrics.jsonl").exists():
            print(
                f"Warning: metrics not found in {run_dir}",
                file=sys.stderr,
            )
            continue
        try:
            rows = _load_metrics_fast(run_dir, step=downsample_step)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"Warning: skipping seed {item.get('seed')} due to "
                f"unreadable metrics: {exc}",
                file=sys.stderr,
            )
            continue
        metrics: list[dict] = []
        for row in rows:
            if "episodes" not in row or "success_rate" not in row:
                continue
            metrics.append(
                {
                    "episodes": int(row["episodes"]),
                    "success_rate": float(row["success_rate"]),
                }
            )
        metrics_by_seed.append(metrics)
    return metrics_by_seed


def _resolve_manifest_key(section: dict, *candidates: str) -> str | None:
    for key in candidates:
        if key in section:
            return key
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute aggregate curve 80%% crossing (what the plot shows)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="plots/requested_experiments_manifest_S0_n10000_w500_seeds30.json",
    )
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = _read_json(manifest_path)
    exp6 = manifest.get("experiment_6", {})
    if not exp6:
        print("Error: experiment_6 not found in manifest", file=sys.stderr)
        sys.exit(1)

    ordered_d = sorted(
        int(k[2:])
        for k in exp6.keys()
        if k.startswith("d=") and k[2:].isdigit()
    )

    # Load summary JSON for comparison (mean of per-seed crossing)
    num_seeds = manifest["config"].get("num_seeds", 30)
    summary_path = manifest_path.parent / f"exp6_degree_sweep_S0_n10000_summary_seeds{num_seeds}.json"
    if not summary_path.exists():
        summary_path = _resolve_path("plots") / f"exp6_degree_sweep_S0_n10000_summary_seeds{num_seeds}.json"
    summary: dict | None = None
    if summary_path.exists():
        summary = _read_json(summary_path)

    print("Experiment 6: Episodes to 80% success rate")
    print("=" * 60)
    print(
        "Aggregate crossing = episode where mean(success_rate across seeds) >= 0.8"
    )
    print("(This is what the plot shows.)")
    print()
    print(
        "Mean per-seed     = mean of each seed's first episode reaching 0.8"
    )
    print("(This is what the JSON episodes_to_80_reached_only reports.)")
    print()

    results: list[tuple[str, int | None, float | None]] = []

    for d in ordered_d:
        display_label = f"d={d}"
        manifest_key = _resolve_manifest_key(exp6, display_label)
        if manifest_key is None:
            continue

        seed_records = exp6[manifest_key]
        seed_metrics = _load_seed_metrics(seed_records)

        if not seed_metrics:
            print(f"{display_label}: No metrics loaded (run data missing?)")
            results.append((display_label, None, None))
            continue

        agg = _aggregate_curve(seed_metrics, max_points=args.max_points)
        if agg is None:
            print(f"{display_label}: Could not compute aggregate curve")
            results.append((display_label, None, None))
            continue

        x_grid, mean_curve = agg
        crossing = _find_aggregate_crossing(
            x_grid, mean_curve, threshold=args.threshold
        )

        mean_per_seed: float | None = None
        if summary and display_label in summary:
            agg_data = summary[display_label].get("aggregate", {})
            et80 = agg_data.get("episodes_to_80_reached_only", {})
            if "mean" in et80:
                mean_per_seed = float(et80["mean"])

        results.append((display_label, crossing, mean_per_seed))

        if crossing is not None:
            mean_str = f"{mean_per_seed:,.0f}" if mean_per_seed is not None else "N/A"
            print(
                f"  {display_label}:  aggregate crossing = {crossing:,}  |  "
                f"mean per-seed = {mean_str}"
            )
        else:
            print(f"  {display_label}:  aggregate never reached {args.threshold}")

    print()
    print("For the blog (matching the plot):")
    print("-" * 40)
    for label, crossing, _ in results:
        if crossing is not None:
            print(f"  {label}: {crossing:,}")
        else:
            print(f"  {label}: (no crossing)")

    if any(r[1] is None for r in results):
        print()
        print(
            "Note: Some runs have no metrics. Ensure v2/runs/ contains the "
            "exp6 run directories with metrics.jsonl from the 30-seed experiment."
        )


if __name__ == "__main__":
    main()
