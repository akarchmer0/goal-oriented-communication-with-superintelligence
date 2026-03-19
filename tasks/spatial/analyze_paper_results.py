"""
Post-processing script for paper experiments.
Reads summary.json from each completed run, extracts the correct metrics,
and generates tables/figures for the paper.
"""

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_FIELDS = {
    "rl_hidden": "final_objective_value",
    "gd": "final_baseline_objective_value",
    "adam": "final_adam_baseline_objective_value",
    "rl_no_oracle": "final_no_oracle_objective_value",
    "rl_visible": "final_visible_gradient_objective_value",
    "distance": "final_distance_to_ref_value",
}


def read_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "summary.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    with Path(manifest_path).open() as f:
        return json.load(f)


def ci95(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)])
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def stats(values: list[float]) -> dict[str, float]:
    arr = np.array([v for v in values if np.isfinite(v)])
    if len(arr) == 0:
        return {"mean": float("nan"), "ci95": 0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "ci95": ci95(values),
        "n": len(arr),
    }


def fmt(val: float, ci: float = 0) -> str:
    if not np.isfinite(val):
        return "---"
    if ci > 0:
        return f"{val:.4f} ± {ci:.4f}"
    return f"{val:.4f}"


# ============================================================
# Experiment 1: Phase Transition
# ============================================================
def analyze_exp1(manifest_path: str = "plots/paper_exp1_phase_transition/manifest.json"):
    manifest = read_manifest(manifest_path)
    runs = manifest.get("runs", {})

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: HIDDEN DIMENSION PHASE TRANSITION")
    print("=" * 70)

    for K_val in [3, 1]:
        k_runs = {k: v for k, v in runs.items() if v.get("basis_complexity") == K_val}
        if not k_runs:
            continue

        by_D: dict[int, list[dict]] = {}
        for run in k_runs.values():
            run_dir = run.get("run_dir", "")
            summary = read_summary(run_dir)
            D = run["hidden_dim"]
            by_D.setdefault(D, []).append(summary)

        print(f"\nK = {K_val}:")
        print(f"{'D':>6} | {'RL Hidden':>20} | {'GD':>20} | {'Adam':>20} | {'RL No Oracle':>20}")
        print("-" * 95)
        for D in sorted(by_D.keys()):
            summaries = by_D[D]
            rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
            gd = stats([s.get("final_baseline_objective_value", float("nan")) for s in summaries])
            adam = stats([s.get("final_adam_baseline_objective_value", float("nan")) for s in summaries])
            no_or = stats([s.get("final_no_oracle_objective_value", float("nan")) for s in summaries])
            print(f"{D:>6} | {fmt(rl['mean'], rl['ci95']):>20} | "
                  f"{fmt(gd['mean'], gd['ci95']):>20} | "
                  f"{fmt(adam['mean'], adam['ci95']):>20} | "
                  f"{fmt(no_or['mean'], no_or['ci95']):>20}")


# ============================================================
# Experiment 2: Basis Complexity
# ============================================================
def analyze_exp2(manifest_path: str = "plots/paper_exp2_basis_complexity/manifest.json"):
    manifest = read_manifest(manifest_path)
    runs = manifest.get("runs", {})
    predictions = manifest.get("predictions", {})

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: BASIS COMPLEXITY SWEEP")
    print("=" * 70)

    fixed_runs = {k: v for k, v in runs.items() if v.get("arm") == "fixed_D"}
    by_K: dict[int, list[dict]] = {}
    for run in fixed_runs.values():
        summary = read_summary(run.get("run_dir", ""))
        by_K.setdefault(run["K"], []).append(summary)

    print(f"\nFixed D = {manifest.get('fixed_D', 200)}:")
    print(f"{'K':>4} | {'D*':>8} | {'Coverage':>8} | {'RL Hidden':>20} | {'GD':>20} | {'Adam':>20}")
    print("-" * 80)
    for K in sorted(by_K.keys()):
        summaries = by_K[K]
        pred = predictions.get(str(K), {})
        rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
        gd = stats([s.get("final_baseline_objective_value", float("nan")) for s in summaries])
        adam = stats([s.get("final_adam_baseline_objective_value", float("nan")) for s in summaries])
        d_star = pred.get("d_star", 0)
        coverage = pred.get("coverage_at_fixed_D", 0)
        print(f"{K:>4} | {d_star:>8.0f} | {100*coverage:>7.0f}% | "
              f"{fmt(rl['mean'], rl['ci95']):>20} | "
              f"{fmt(gd['mean'], gd['ci95']):>20} | "
              f"{fmt(adam['mean'], adam['ci95']):>20}")


# ============================================================
# Experiment 3: Bandwidth
# ============================================================
def analyze_exp3(manifest_path: str = "plots/paper_exp3_bandwidth/manifest.json"):
    manifest = read_manifest(manifest_path)
    runs = manifest.get("runs", {})

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: COMMUNICATION BANDWIDTH")
    print("=" * 70)

    # Linear embedding runs
    linear_runs = {k: v for k, v in runs.items() if v.get("method") == "linear_embedding"}
    by_T: dict[int, list[dict]] = {}
    for run in linear_runs.values():
        summary = read_summary(run.get("run_dir", ""))
        by_T.setdefault(run["token_dim"], []).append(summary)

    # Controls
    control_summaries: dict[str, list[dict]] = {}
    for run in runs.values():
        method = run.get("method", "")
        if method in ("convex_gradient", "visible_gradient", "no_oracle"):
            summary = read_summary(run.get("run_dir", ""))
            control_summaries.setdefault(method, []).append(summary)

    print(f"\nLinear embedding (random projection):")
    print(f"{'T':>6} | {'Compression':>12} | {'RL Objective':>20}")
    print("-" * 45)
    for T in sorted(by_T.keys()):
        summaries = by_T[T]
        rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
        ratio = T / manifest.get("hidden_dim", 200)
        print(f"{T:>6} | {ratio:>11.1%} | {fmt(rl['mean'], rl['ci95']):>20}")

    print(f"\nControls:")
    labels = {"convex_gradient": "Full hidden grad", "visible_gradient": "Visible grad", "no_oracle": "No oracle"}
    for method, label in labels.items():
        if method in control_summaries:
            rl = stats([s.get("final_objective_value", float("nan")) for s in control_summaries[method]])
            print(f"  {label:>20}: {fmt(rl['mean'], rl['ci95'])}")


# ============================================================
# Experiment 4: Noise Robustness
# ============================================================
def analyze_exp4(manifest_path: str = "plots/paper_exp4_noise/manifest.json"):
    manifest = read_manifest(manifest_path)
    runs = manifest.get("runs", {})

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: ORACLE NOISE ROBUSTNESS")
    print("=" * 70)

    by_sigma: dict[float, list[dict]] = {}
    for run in runs.values():
        summary = read_summary(run.get("run_dir", ""))
        by_sigma.setdefault(run["noise_std"], []).append(summary)

    print(f"{'σ':>8} | {'RL Hidden':>20} | {'GD':>20} | {'Adam':>20}")
    print("-" * 75)
    for sigma in sorted(by_sigma.keys()):
        summaries = by_sigma[sigma]
        rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
        gd = stats([s.get("final_baseline_objective_value", float("nan")) for s in summaries])
        adam = stats([s.get("final_adam_baseline_objective_value", float("nan")) for s in summaries])
        print(f"{sigma:>8.1f} | {fmt(rl['mean'], rl['ci95']):>20} | "
              f"{fmt(gd['mean'], gd['ci95']):>20} | "
              f"{fmt(adam['mean'], adam['ci95']):>20}")


# ============================================================
# Experiment 5: Transfer
# ============================================================
def analyze_exp5(manifest_path: str = "plots/paper_exp5_transfer/manifest.json"):
    manifest = read_manifest(manifest_path)
    runs = manifest.get("runs", {})
    evals = manifest.get("evaluations", {})

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: MAP FAMILY TRANSFER")
    print("=" * 70)

    # Training results
    print("\nTraining objectives (same family):")
    for family in ["FOURIER", "MLP"]:
        family_runs = [v for v in runs.values() if v.get("train_family") == family]
        summaries = [read_summary(r.get("run_dir", "")) for r in family_runs]
        rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
        print(f"  {family:>10}: {fmt(rl['mean'], rl['ci95'])}")

    # Cross-evaluation results
    if evals:
        print("\nTransfer matrix (mean objective):")
        families = manifest.get("families", ["FOURIER", "MLP"])
        print(f"{'Train \\ Eval':>15} | ", end="")
        for f in families:
            print(f"{f:>12} | ", end="")
        print()
        print("-" * 45)
        for train_f in families:
            print(f"{train_f:>15} | ", end="")
            for eval_f in families:
                matching = [v for v in evals.values()
                           if v.get("train_family") == train_f and v.get("eval_family") == eval_f]
                objs = [e.get("mean_objective", float("nan")) for e in matching]
                s = stats(objs)
                print(f"{fmt(s['mean']):>12} | ", end="")
            print()


# ============================================================
# Experiment 6: Scaling
# ============================================================
def analyze_exp6(manifest_path: str = "plots/paper_exp6_scaling/manifest.json"):
    manifest = read_manifest(manifest_path)
    runs = manifest.get("runs", {})
    dim_grid = manifest.get("dim_grid", {})

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: VISIBLE DIMENSION SCALING")
    print("=" * 70)

    visible_dims = manifest.get("visible_dims", [])

    print(f"{'d':>4} | {'D':>6} | {'RL Hidden':>20} | {'RL No Oracle':>20} | {'GD':>20} | {'Adam':>20}")
    print("-" * 95)
    for d in sorted(visible_dims):
        info = dim_grid.get(str(d), {})
        D = info.get("D", "?")

        hidden_runs = [v for v in runs.values() if v.get("d") == d and v.get("oracle") == "hidden"]
        no_oracle_runs = [v for v in runs.values() if v.get("d") == d and v.get("oracle") == "no_oracle"]

        h_summaries = [read_summary(r.get("run_dir", "")) for r in hidden_runs]
        n_summaries = [read_summary(r.get("run_dir", "")) for r in no_oracle_runs]

        rl_h = stats([s.get("final_objective_value", float("nan")) for s in h_summaries])
        rl_n = stats([s.get("final_objective_value", float("nan")) for s in n_summaries])
        gd = stats([s.get("final_baseline_objective_value", float("nan")) for s in h_summaries])
        adam = stats([s.get("final_adam_baseline_objective_value", float("nan")) for s in h_summaries])

        print(f"{d:>4} | {D:>6} | {fmt(rl_h['mean'], rl_h['ci95']):>20} | "
              f"{fmt(rl_n['mean'], rl_n['ci95']):>20} | "
              f"{fmt(gd['mean'], gd['ci95']):>20} | "
              f"{fmt(adam['mean'], adam['ci95']):>20}")


# ============================================================
# Generate paper plots
# ============================================================
def make_paper_plots():
    """Generate publication-quality plots for all experiments."""
    plots_dir = Path("plots/paper_figures")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })

    # --- Exp 1: Phase Transition ---
    try:
        manifest = read_manifest("plots/paper_exp1_phase_transition/manifest.json")
        runs = manifest.get("runs", {})

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for K_val, ax, color in [(3, axes[0], "steelblue"), (1, axes[1], "coral")]:
            k_runs = {k: v for k, v in runs.items() if v.get("basis_complexity") == K_val}
            by_D: dict[int, list[dict]] = {}
            for run in k_runs.values():
                summary = read_summary(run.get("run_dir", ""))
                by_D.setdefault(run["hidden_dim"], []).append(summary)

            Ds = sorted(by_D.keys())
            rl_means, rl_cis = [], []
            gd_means, gd_cis = [], []
            adam_means, adam_cis = [], []

            for D in Ds:
                summaries = by_D[D]
                rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
                gd = stats([s.get("final_baseline_objective_value", float("nan")) for s in summaries])
                adam = stats([s.get("final_adam_baseline_objective_value", float("nan")) for s in summaries])
                rl_means.append(rl["mean"]); rl_cis.append(rl["ci95"])
                gd_means.append(gd["mean"]); gd_cis.append(gd["ci95"])
                adam_means.append(adam["mean"]); adam_cis.append(adam["ci95"])

            from .run_paper_exp1_phase_transition import _coupon_collector_threshold
            d_star = _coupon_collector_threshold(K_val, manifest.get("visible_dim", 2))

            ax.errorbar(Ds, rl_means, yerr=rl_cis, marker="o", capsize=3, linewidth=2, color="green", label="RL (hidden grad)")
            ax.errorbar(Ds, gd_means, yerr=gd_cis, marker="^", capsize=3, linewidth=1.5, color="gray", linestyle="--", label="GD")
            ax.errorbar(Ds, adam_means, yerr=adam_cis, marker="s", capsize=3, linewidth=1.5, color="orange", linestyle="--", label="Adam")
            ax.axvline(d_star, color="red", linestyle=":", alpha=0.6, linewidth=1.5, label=f"$D^* = {d_star:.0f}$")
            ax.set_xlabel("Hidden dimension $D$")
            ax.set_ylabel("Final objective")
            ax.set_title(f"$K = {K_val}$, $d = 2$")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_ylim(bottom=-0.02)

        fig.suptitle("Hidden Dimension Phase Transition", fontsize=15, y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "fig1_phase_transition.pdf")
        fig.savefig(plots_dir / "fig1_phase_transition.png", dpi=200)
        plt.close(fig)
        print(f"Saved fig1_phase_transition")
    except Exception as e:
        print(f"Exp1 plot skipped: {e}")

    # --- Exp 2: Basis Complexity ---
    try:
        manifest = read_manifest("plots/paper_exp2_basis_complexity/manifest.json")
        runs = manifest.get("runs", {})
        predictions = manifest.get("predictions", {})

        fixed_runs = {k: v for k, v in runs.items() if v.get("arm") == "fixed_D"}
        by_K: dict[int, list[dict]] = {}
        for run in fixed_runs.values():
            summary = read_summary(run.get("run_dir", ""))
            by_K.setdefault(run["K"], []).append(summary)

        Ks = sorted(by_K.keys())
        rl_means, rl_cis = [], []
        gd_means, gd_cis = [], []

        for K in Ks:
            summaries = by_K[K]
            rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
            gd = stats([s.get("final_baseline_objective_value", float("nan")) for s in summaries])
            rl_means.append(rl["mean"]); rl_cis.append(rl["ci95"])
            gd_means.append(gd["mean"]); gd_cis.append(gd["ci95"])

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.errorbar(Ks, rl_means, yerr=rl_cis, marker="o", capsize=3, linewidth=2, color="green", label="RL (hidden grad)")
        ax1.errorbar(Ks, gd_means, yerr=gd_cis, marker="^", capsize=3, linewidth=1.5, color="gray", linestyle="--", label="GD")
        ax1.set_xlabel("Basis complexity $K$")
        ax1.set_ylabel("Final objective")
        ax1.set_title(f"Basis Complexity Sweep ($D = {manifest.get('fixed_D', 200)}$, $d = 2$)")
        ax1.legend()
        ax1.grid(True, alpha=0.2)

        # Secondary axis: D*/D ratio
        ax2 = ax1.twinx()
        ratios = [predictions.get(str(K), {}).get("d_star", 0) / manifest.get("fixed_D", 200) for K in Ks]
        ax2.plot(Ks, ratios, marker="D", linestyle=":", color="red", alpha=0.5, label="$D^*/D$")
        ax2.axhline(1.0, color="red", linestyle=":", alpha=0.3)
        ax2.set_ylabel("$D^*/D$", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        fig.tight_layout()
        fig.savefig(plots_dir / "fig2_basis_complexity.pdf")
        fig.savefig(plots_dir / "fig2_basis_complexity.png", dpi=200)
        plt.close(fig)
        print(f"Saved fig2_basis_complexity")
    except Exception as e:
        print(f"Exp2 plot skipped: {e}")

    # --- Exp 3: Bandwidth ---
    try:
        manifest = read_manifest("plots/paper_exp3_bandwidth/manifest.json")
        runs = manifest.get("runs", {})

        linear_runs = {k: v for k, v in runs.items() if v.get("method") == "linear_embedding"}
        by_T: dict[int, list[dict]] = {}
        for run in linear_runs.values():
            summary = read_summary(run.get("run_dir", ""))
            by_T.setdefault(run["token_dim"], []).append(summary)

        Ts = sorted(by_T.keys())
        means, cis_vals = [], []
        for T in Ts:
            summaries = by_T[T]
            s = stats([s.get("final_objective_value", float("nan")) for s in summaries])
            means.append(s["mean"]); cis_vals.append(s["ci95"])

        # Controls
        controls = {}
        for method_name in ["convex_gradient", "visible_gradient", "no_oracle"]:
            method_runs = [v for v in runs.values() if v.get("method") == method_name]
            if method_runs:
                summaries = [read_summary(r.get("run_dir", "")) for r in method_runs]
                s = stats([s.get("final_objective_value", float("nan")) for s in summaries])
                controls[method_name] = s["mean"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(Ts, means, yerr=cis_vals, marker="o", capsize=3, linewidth=2.5, color="steelblue", label="Linear embedding", zorder=5)

        if "convex_gradient" in controls:
            ax.axhline(controls["convex_gradient"], color="green", linestyle="--", alpha=0.7, label="Full hidden grad")
        if "visible_gradient" in controls:
            ax.axhline(controls["visible_gradient"], color="orange", linestyle="-.", alpha=0.7, label="Visible grad")
        if "no_oracle" in controls:
            ax.axhline(controls["no_oracle"], color="red", linestyle=":", alpha=0.7, label="No oracle")

        ax.set_xscale("log")
        ax.set_xlabel("Token dimension (bandwidth)")
        ax.set_ylabel("Final objective")
        ax.set_title("Communication Bandwidth")
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / "fig3_bandwidth.pdf")
        fig.savefig(plots_dir / "fig3_bandwidth.png", dpi=200)
        plt.close(fig)
        print(f"Saved fig3_bandwidth")
    except Exception as e:
        print(f"Exp3 plot skipped: {e}")

    # --- Exp 4: Noise ---
    try:
        manifest = read_manifest("plots/paper_exp4_noise/manifest.json")
        runs = manifest.get("runs", {})

        by_sigma: dict[float, list[dict]] = {}
        for run in runs.values():
            summary = read_summary(run.get("run_dir", ""))
            by_sigma.setdefault(run["noise_std"], []).append(summary)

        sigmas = sorted(by_sigma.keys())
        rl_means, rl_cis = [], []
        gd_means, gd_cis = [], []

        for sigma in sigmas:
            summaries = by_sigma[sigma]
            rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
            gd = stats([s.get("final_baseline_objective_value", float("nan")) for s in summaries])
            rl_means.append(rl["mean"]); rl_cis.append(rl["ci95"])
            gd_means.append(gd["mean"]); gd_cis.append(gd["ci95"])

        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = [max(s, 0.05) for s in sigmas]
        ax.errorbar(x_pos, rl_means, yerr=rl_cis, marker="o", capsize=3, linewidth=2.5, color="green", label="RL (hidden grad)")
        ax.errorbar(x_pos, gd_means, yerr=gd_cis, marker="^", capsize=3, linewidth=1.5, color="gray", linestyle="--", label="GD (unaffected)")
        ax.set_xscale("log")
        ax.axvline(1.0, color="red", linestyle=":", alpha=0.5, label="SNR $\\approx$ 1")
        ax.set_xlabel("Oracle noise $\\sigma$")
        ax.set_ylabel("Final objective")
        ax.set_title("Oracle Noise Robustness")
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / "fig4_noise.pdf")
        fig.savefig(plots_dir / "fig4_noise.png", dpi=200)
        plt.close(fig)
        print(f"Saved fig4_noise")
    except Exception as e:
        print(f"Exp4 plot skipped: {e}")

    # --- Exp 6: Scaling ---
    try:
        manifest = read_manifest("plots/paper_exp6_scaling/manifest.json")
        runs = manifest.get("runs", {})
        dim_grid = manifest.get("dim_grid", {})
        visible_dims = sorted(manifest.get("visible_dims", []))

        rl_means, rl_cis = [], []
        gd_means, gd_cis = [], []
        adam_means, adam_cis = [], []

        for d in visible_dims:
            hidden_runs = [v for v in runs.values() if v.get("d") == d and v.get("oracle") == "hidden"]
            summaries = [read_summary(r.get("run_dir", "")) for r in hidden_runs]
            rl = stats([s.get("final_objective_value", float("nan")) for s in summaries])
            gd = stats([s.get("final_baseline_objective_value", float("nan")) for s in summaries])
            adam = stats([s.get("final_adam_baseline_objective_value", float("nan")) for s in summaries])
            rl_means.append(rl["mean"]); rl_cis.append(rl["ci95"])
            gd_means.append(gd["mean"]); gd_cis.append(gd["ci95"])
            adam_means.append(adam["mean"]); adam_cis.append(adam["ci95"])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(visible_dims, rl_means, yerr=rl_cis, marker="o", capsize=3, linewidth=2.5, color="green", label="RL (hidden grad)")
        ax.errorbar(visible_dims, gd_means, yerr=gd_cis, marker="^", capsize=3, linewidth=1.5, color="gray", linestyle="--", label="GD")
        ax.errorbar(visible_dims, adam_means, yerr=adam_cis, marker="s", capsize=3, linewidth=1.5, color="orange", linestyle="--", label="Adam")
        ax.set_xlabel("Visible dimension $d$")
        ax.set_ylabel("Final objective")
        ax.set_title("Scaling with Visible Dimension ($K = 1$)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.set_xticks(visible_dims)
        fig.tight_layout()
        fig.savefig(plots_dir / "fig5_scaling.pdf")
        fig.savefig(plots_dir / "fig5_scaling.png", dpi=200)
        plt.close(fig)
        print(f"Saved fig5_scaling")
    except Exception as e:
        print(f"Exp6 plot skipped: {e}")


def main():
    print("=" * 70)
    print("PAPER EXPERIMENT RESULTS ANALYSIS")
    print("=" * 70)

    for exp_num, func, path in [
        (1, analyze_exp1, "plots/paper_exp1_phase_transition/manifest.json"),
        (2, analyze_exp2, "plots/paper_exp2_basis_complexity/manifest.json"),
        (3, analyze_exp3, "plots/paper_exp3_bandwidth/manifest.json"),
        (4, analyze_exp4, "plots/paper_exp4_noise/manifest.json"),
        (5, analyze_exp5, "plots/paper_exp5_transfer/manifest.json"),
        (6, analyze_exp6, "plots/paper_exp6_scaling/manifest.json"),
    ]:
        if Path(path).exists():
            try:
                func(path)
            except Exception as e:
                print(f"\nExp {exp_num} analysis failed: {e}")
        else:
            print(f"\nExp {exp_num}: manifest not found at {path}")

    print("\n\nGenerating paper plots...")
    make_paper_plots()
    print("\nDone!")


if __name__ == "__main__":
    main()
