#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np

from v2.plotting import plot_spatial_optimization_curve_summary


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

METHOD_COLUMNS = {
    "gd": ("avg_baseline_final_objective", "baseline_final_objective"),
    "adam": ("avg_adam_baseline_final_objective", "adam_baseline_final_objective"),
    "rl_no_oracle": ("avg_no_oracle_final_objective", "no_oracle_final_objective"),
    "rl_visible_oracle": (
        "avg_visible_gradient_final_objective",
        "avg_visible_oracle_final_objective",
        "visible_gradient_final_objective",
        "visible_oracle_final_objective",
    ),
    "rl_hidden_gradient": ("avg_final_objective", "final_objective"),
}


def _parse_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _resolve_metric_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    candidates = (
        resolved / "metrics.csv",
        resolved / "metrics.jsonl",
        resolved / "metrics.json",
        resolved / "spatial_optimization_curves.json",
        resolved / "spatial_optimization_curves_over_seeds.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find metrics data in directory: {resolved} "
        "(expected metrics.csv, metrics.jsonl, metrics.json, or spatial_optimization_curves*.json)"
    )


def _extract_from_metric_rows(rows: list[dict]) -> dict[str, list[np.ndarray]]:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if not valid_rows:
        return {}

    if all("episodes" in row for row in valid_rows):
        valid_rows = sorted(valid_rows, key=lambda item: _parse_float(item.get("episodes")))

    curves: dict[str, list[np.ndarray]] = {}
    for method_key in METHOD_ORDER:
        values: list[float] = []
        cols = METHOD_COLUMNS[method_key]
        for row in valid_rows:
            selected = float("nan")
            for col in cols:
                candidate = _parse_float(row.get(col))
                if np.isfinite(candidate):
                    selected = candidate
                    break
            values.append(selected)
        arr = np.asarray(values, dtype=np.float64)
        if arr.size >= 2 and np.any(np.isfinite(arr)):
            curves[method_key] = [arr]
    return curves


def _extract_from_spatial_json(payload: dict) -> dict[str, list[np.ndarray]]:
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        return {}

    curves: dict[str, list[np.ndarray]] = {}
    for method_key in METHOD_ORDER:
        method_payload = methods.get(method_key)
        if not isinstance(method_payload, dict):
            continue

        samples: list[np.ndarray] = []
        task_curves = method_payload.get("task_curves")
        if isinstance(task_curves, list):
            arr = np.asarray(task_curves, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                for row in arr:
                    if np.any(np.isfinite(row)):
                        samples.append(row.astype(np.float64))

        if not samples:
            mean_curve = method_payload.get("mean_curve")
            if isinstance(mean_curve, list):
                arr = np.asarray(mean_curve, dtype=np.float64).reshape(-1)
                if arr.size >= 2 and np.any(np.isfinite(arr)):
                    samples.append(arr)

        if samples:
            curves[method_key] = samples
    return curves


def _load_curves_from_file(path: Path) -> dict[str, list[np.ndarray]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(path)
        return _extract_from_metric_rows(rows)
    if suffix == ".jsonl":
        rows = _read_jsonl(path)
        return _extract_from_metric_rows(rows)
    if suffix == ".json":
        payload = _read_json(path)
        if isinstance(payload, dict):
            spatial_curves = _extract_from_spatial_json(payload)
            if spatial_curves:
                return spatial_curves
            rows = payload.get("metrics")
            if isinstance(rows, list):
                row_dicts = [item for item in rows if isinstance(item, dict)]
                return _extract_from_metric_rows(row_dicts)
        if isinstance(payload, list):
            row_dicts = [item for item in payload if isinstance(item, dict)]
            return _extract_from_metric_rows(row_dicts)
    raise ValueError(f"Unsupported or unreadable metrics file: {path}")


def _merge_curves(
    per_file_curves: list[dict[str, list[np.ndarray]]],
    max_steps: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    merged_samples: dict[str, list[np.ndarray]] = {key: [] for key in METHOD_ORDER}
    for entry in per_file_curves:
        for method_key, samples in entry.items():
            merged_samples.setdefault(method_key, []).extend(samples)

    mean_curves: dict[str, np.ndarray] = {}
    std_curves: dict[str, np.ndarray] = {}

    for method_key in METHOD_ORDER:
        samples = merged_samples.get(method_key, [])
        clipped: list[np.ndarray] = []
        for sample in samples:
            arr = np.asarray(sample, dtype=np.float64).reshape(-1)
            if max_steps is not None:
                arr = arr[:max_steps]
            if arr.size >= 2 and np.any(np.isfinite(arr)):
                clipped.append(arr)
        if not clipped:
            continue

        min_len = min(arr.size for arr in clipped)
        if min_len < 2:
            continue

        stacked = np.stack([arr[:min_len] for arr in clipped], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(invalid="ignore"):
                mean_curve = np.nanmean(stacked, axis=0)
                std_curve = np.nanstd(stacked, axis=0)
        if not np.any(np.isfinite(mean_curve)):
            continue
        mean_curves[method_key] = mean_curve.astype(np.float64)
        std_curves[method_key] = std_curve.astype(np.float64)

    return mean_curves, std_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute spatial optimization curves (mean ± std) from metrics CSV/JSONL/JSON files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "Metrics sources: files (.csv/.jsonl/.json) or run directories "
            "(auto-resolves metrics file)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spatial_optimization_curves_mean_std_recomputed.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of optimization steps to plot on the x-axis.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Spatial optimization mean ± std (recomputed)",
        help="Plot title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps is not None and args.max_steps < 2:
        raise ValueError("--max-steps must be >= 2 when provided.")

    resolved_files = [_resolve_metric_file(path) for path in args.inputs]
    per_file_curves = [_load_curves_from_file(path) for path in resolved_files]

    mean_curves, std_curves = _merge_curves(per_file_curves, args.max_steps)
    if not mean_curves:
        raise ValueError(
            "No valid optimization curves found. "
            "Check that your metrics files include objective columns."
        )

    output_path = args.output.expanduser().resolve()
    plot_spatial_optimization_curve_summary(
        method_mean_curves=mean_curves,
        method_std_curves=std_curves,
        output_path=output_path,
        title=args.title,
        method_labels=METHOD_LABELS,
        y_label="Normalized objective E(F(z))",
        x_label="Optimization step",
    )

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "input_files": [str(path) for path in resolved_files],
                "max_steps": args.max_steps,
                "methods_plotted": [key for key in METHOD_ORDER if key in mean_curves],
                "num_sample_curves": {
                    key: int(
                        sum(
                            len(entry.get(key, []))
                            for entry in per_file_curves
                        )
                    )
                    for key in METHOD_ORDER
                    if key in mean_curves
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
