import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep matplotlib fully headless and in writable cache locations.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from .run_optimizer_studies import (
    _aggregate_meta_results,
    _aggregate_search_results,
    _plot_meta_curves,
    _plot_meta_final_summary,
    _plot_search_budget_curve,
    _plot_search_wall_clock_curve,
    _plot_search_max_budget_summary,
)


def _resolve_output_path(raw: str | None, *, manifest_dir: Path, fallback: Path) -> Path:
    if raw is None or str(raw).strip() == "":
        return fallback.resolve()
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_dir / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _collect_meta_seed_payloads(meta_section: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for run_entry in meta_section.get("runs", []):
        payload_path_str = run_entry.get("meta_evaluation_json")
        if not payload_path_str:
            continue
        payload_path = Path(str(payload_path_str)).expanduser().resolve()
        if not payload_path.exists():
            print(f"[warn] Missing meta seed evaluation JSON: {payload_path}")
            continue
        try:
            payload = _read_json(payload_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] Skipping unreadable meta seed JSON {payload_path}: {exc}")
            continue
        payloads.append(payload)
    return payloads


def _collect_search_seed_payloads(search_section: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for run_entry in search_section.get("runs", []):
        payload_path_str = run_entry.get("search_evaluation_json")
        if not payload_path_str:
            continue
        payload_path = Path(str(payload_path_str)).expanduser().resolve()
        if not payload_path.exists():
            print(f"[warn] Missing search seed evaluation JSON: {payload_path}")
            continue
        try:
            payload = _read_json(payload_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] Skipping unreadable search seed JSON {payload_path}: {exc}")
            continue
        payloads.append(payload)
    return payloads


def _regenerate_meta(
    *,
    meta_section: dict[str, Any],
    plots_root: Path,
    plot_data_root: Path,
) -> dict[str, Any]:
    plots_root.mkdir(parents=True, exist_ok=True)
    plot_data_root.mkdir(parents=True, exist_ok=True)

    seed_payloads = _collect_meta_seed_payloads(meta_section)
    if not seed_payloads:
        return {
            "enabled": False,
            "reason": "No readable per-seed meta evaluation JSON files found.",
        }

    aggregate = _aggregate_meta_results(seed_payloads)
    aggregate_path = plot_data_root / "meta_optimizer_aggregate.json"
    _write_json(aggregate_path, aggregate)

    curves_plot = plots_root / "meta_optimizer_objective_vs_step.png"
    final_plot = plots_root / "meta_optimizer_final_objective_summary.png"
    _plot_meta_curves(aggregate=aggregate, output_path=curves_plot)
    _plot_meta_final_summary(aggregate=aggregate, output_path=final_plot)

    plots_payload: dict[str, str] = {}
    if curves_plot.exists():
        plots_payload["objective_vs_step_plot"] = str(curves_plot.resolve())
    if final_plot.exists():
        plots_payload["final_objective_summary_plot"] = str(final_plot.resolve())

    meta_section["plots_root"] = str(plots_root.resolve())
    meta_section["plot_data_root"] = str(plot_data_root.resolve())
    meta_section["aggregate_json"] = str(aggregate_path.resolve())
    meta_section["plots"] = plots_payload
    meta_section["last_plotted_at_utc"] = datetime.now(timezone.utc).isoformat()

    return {
        "enabled": True,
        "num_seed_payloads": int(len(seed_payloads)),
        "aggregate_json": str(aggregate_path.resolve()),
        "plots": plots_payload,
    }


def _regenerate_search(
    *,
    search_section: dict[str, Any],
    plots_root: Path,
    plot_data_root: Path,
) -> dict[str, Any]:
    plots_root.mkdir(parents=True, exist_ok=True)
    plot_data_root.mkdir(parents=True, exist_ok=True)

    seed_payloads = _collect_search_seed_payloads(search_section)
    if not seed_payloads:
        return {
            "enabled": False,
            "reason": "No readable per-seed search evaluation JSON files found.",
        }

    aggregate = _aggregate_search_results(seed_payloads)
    aggregate_path = plot_data_root / "search_algorithm_aggregate.json"
    _write_json(aggregate_path, aggregate)

    budget_curve_plot = plots_root / "search_algorithm_budget_curve.png"
    wall_clock_curve_plot = plots_root / "search_algorithm_wall_clock_curve.png"
    max_budget_plot = plots_root / "search_algorithm_max_budget_summary.png"
    _plot_search_budget_curve(aggregate=aggregate, output_path=budget_curve_plot)
    _plot_search_wall_clock_curve(aggregate=aggregate, output_path=wall_clock_curve_plot)
    _plot_search_max_budget_summary(aggregate=aggregate, output_path=max_budget_plot)

    plots_payload: dict[str, str] = {}
    if budget_curve_plot.exists():
        plots_payload["budget_curve_plot"] = str(budget_curve_plot.resolve())
    if wall_clock_curve_plot.exists():
        plots_payload["wall_clock_curve_plot"] = str(wall_clock_curve_plot.resolve())
    if max_budget_plot.exists():
        plots_payload["max_budget_summary_plot"] = str(max_budget_plot.resolve())

    search_section["plots_root"] = str(plots_root.resolve())
    search_section["plot_data_root"] = str(plot_data_root.resolve())
    search_section["aggregate_json"] = str(aggregate_path.resolve())
    search_section["plots"] = plots_payload
    search_section["last_plotted_at_utc"] = datetime.now(timezone.utc).isoformat()

    return {
        "enabled": True,
        "num_seed_payloads": int(len(seed_payloads)),
        "aggregate_json": str(aggregate_path.resolve()),
        "plots": plots_payload,
    }


def plot_from_manifest(
    *,
    manifest_path: Path,
    output_root_override: str | None = None,
    write_manifest: bool = True,
    write_summary: bool = True,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = _read_json(manifest_path)

    suite_root = _resolve_output_path(
        manifest.get("suite_root"),
        manifest_dir=manifest_dir,
        fallback=manifest_dir,
    )
    if output_root_override is not None and str(output_root_override).strip():
        output_root = _resolve_output_path(
            output_root_override,
            manifest_dir=manifest_dir,
            fallback=suite_root,
        )
    else:
        output_root = suite_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest["suite_root"] = str(suite_root.resolve())
    manifest["last_plotted_at_utc"] = datetime.now(timezone.utc).isoformat()

    meta_result = {"enabled": False, "reason": "meta_optimizer section missing"}
    search_result = {"enabled": False, "reason": "search_algorithm section missing"}

    meta_section = manifest.get("meta_optimizer")
    if isinstance(meta_section, dict):
        meta_result = _regenerate_meta(
            meta_section=meta_section,
            plots_root=output_root / "meta_optimizer" / "plots",
            plot_data_root=output_root / "meta_optimizer" / "plot_data",
        )

    search_section = manifest.get("search_algorithm")
    if isinstance(search_section, dict):
        search_result = _regenerate_search(
            search_section=search_section,
            plots_root=output_root / "search_algorithm" / "plots",
            plot_data_root=output_root / "search_algorithm" / "plot_data",
        )

    if write_manifest:
        _write_json(manifest_path, manifest)

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "suite_root": str(suite_root.resolve()),
        "output_root": str(output_root.resolve()),
        "study_mode": str(manifest.get("study_mode", "unknown")),
        "num_seeds": int(len(manifest.get("seeds", []))),
        "seeds": [int(seed) for seed in manifest.get("seeds", [])],
        "meta_optimizer": meta_result,
        "search_algorithm": search_result,
        "plot_cmd": f"python3 -m tasks.spatial.plot_optimizer_studies --manifest {manifest_path}",
    }

    if write_summary:
        summary_path = suite_root / "suite_summary.json"
        _write_json(summary_path, summary)
        summary["summary_path"] = str(summary_path.resolve())

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate optimizer-study plots from tasks.spatial.run_optimizer_studies manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest.json created by tasks.spatial.run_optimizer_studies.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Optional output root override for regenerated plots/data. "
            "If omitted, uses suite_root from manifest."
        ),
    )
    parser.add_argument(
        "--no_write_manifest",
        action="store_true",
        help="Do not write updated plot/data paths back into the manifest.",
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
        output_root_override=args.output_root,
        write_manifest=not bool(args.no_write_manifest),
        write_summary=not bool(args.no_write_summary),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
