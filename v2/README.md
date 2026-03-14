# Goal-Oriented Communication (CPU-only)

This `v2` prototype implements a CPU-only RL setup for goal-oriented communication with an alien oracle.

It now supports two tasks:
- `graph`: fixed directed graph navigation with discrete ciphers (`fixed_cipher`, `fresh_cipher`, `fst_cipher`)
- `spatial`: convex hidden-space optimization with 2D control and gradient messages (`convex_gradient`)

Use `--task graph` or `--task spatial` in `v2.train`.

## Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r v2/requirements.txt
```

## Spatial High-Dimensional Planning

Recommended baseline matching the new hidden-space narrative:

```bash
python3 -m v2.train \
  --task spatial \
  --oracle_mode convex_gradient \
  --spatial_hidden_dim 10 \
  --spatial_visible_dim 2 \
  --spatial_token_dim 10 \
  --spatial_basis_complexity 4 \
  --spatial_coord_limit 8 \
  --base_lr 1.0 \
  --spatial_success_threshold 0.01 \
  --sensing S0 \
  --train_steps 300000
```

When running `--oracle_mode convex_gradient` with baselines enabled (default), training now also runs a parallel **PPO visible gradient** baseline for direct comparison.

Regenerate the spatial learning curves afterward without retraining:

```bash
python3 -m v2.plot_learning_curves \
  --run_dir v2/runs/<run_name>
```

Run the full spatial ablation suite (oracle-only PPO, no baselines in training):

```bash
python3 -m v2.run_spatial_ablation_suite \
  --seeds 0,1,2,3,4 \
  --suite_output_dir plots/spatial_ablation_suite \
  --logdir v2/runs
```

Regenerate ablation plots later (no retraining), useful for formatting iteration:

```bash
python3 -m v2.plot_ablations \
  --manifest plots/spatial_ablation_suite/manifest.json
```

If you want training only, run the suite with `--skip_plotting` and use `v2.plot_ablations` afterward.

This produces:
- per-run `metrics.jsonl` and `summary.json`
- per-run trajectory JSONL + heatmap JSON (for 2D sweeps)
- suite `manifest.json`
- curve plots (individual and mean) for objective/distance/success rate
- heatmap plots (individual and mean) for oracle-noise / reward-noise / D sweeps
- no heatmaps for the `H` sweep by design
- `H` sweep uses budget scaling by default (`step_size *= sqrt(H/2)`) for fairer cross-dimension comparison

Interpretation:
- hidden objective is convex in `R^D`: `E(s)=0.5||s-s*||^2`
- human controls only `z ∈ R^2`; hidden state is `s=F(z)` with nonlinear sinusoidal basis
- oracle sends the true hidden-space gradient `g_t = s_t - s*` each step
- policy learns `(g_t, z_t) -> a_t` with **continuous 2D actions** (any direction), using reward from energy decrease `E(F(z_t)) - E(F(z_{t+1}))`
- spatial diagnostics now include `2D GD` and `2D Adam` baselines

## Spatial Optimizer Studies

Use this runner to keep the two optimizer studies separate and reproducible:

```bash
python3 -m v2.run_spatial_optimizer_studies \
  --study all \
  --seeds 0,1,2,3,4 \
  --suite_output_dir plots/spatial_optimizer_studies \
  --logdir v2/runs
```

Available studies:

- `meta_optimizer`: fixed Fourier map per seed, random start/target per task, equal optimization horizon across methods (`GD`, `Adam`, `RL no oracle`, `RL visible oracle`, `RL hidden gradient`).
- `search_algorithm`: fixed map + fixed start/target per seed (`--spatial_fixed_start_target` behavior), wall-clock-equalized search budgets, and includes `random_search` baseline.

Useful flags:

- `--meta_num_tasks` and `--meta_eval_horizon` control post-training optimizer evaluation.
- `--search_wallclock_budgets_sec` controls compute-equalized search budgets.
- `--search_eval_horizon` and `--search_max_episodes_per_method` control per-budget search rollouts.
- `--search_rl_deterministic` switches search-policy evaluation from stochastic to deterministic actions.

Outputs are written under:

- `plots/spatial_optimizer_studies/meta_optimizer/...`
- `plots/spatial_optimizer_studies/search_algorithm/...`

Each study writes:

- per-seed evaluation JSON (raw task/episode data)
- aggregate JSON for post-hoc analysis
- summary plots
- top-level `manifest.json` and `suite_summary.json`

Regenerate optimizer-study plots later without retraining:

```bash
python3 -m v2.plot_spatial_optimizer_studies \
  --manifest plots/spatial_optimizer_studies/manifest.json
```

## Requested Experiments Only

This repo supports the requested experiments:

1. Encoding comparison (`fixed_cipher`, `fresh_cipher`, `fst_cipher`, `random_message`, `no_message`)
2. Graph-size scaling (fixed cipher, one curve per `n`, viridis palette)
3. Adversarial fixed cipher with `p_lie in {0.01, 0.1, 0.25, 0.5}`
4. Noisy sensing with `sigma in {0, 0.1, 0.5, 1.0, 2.0}` plus binary signal baseline (no noise)
5. FST state sweep with `k in {1, 2, 3, 4}` (`fst_cipher`)
6. Graph degree sweep with `d in {2, 4, 8, 16}` (one curve per out-degree)

All learning curves are running-average success rate vs episodes.

Default knobs are centralized in `v2/config.py`:
- `RequestedExperimentsConfig` for `v2.run_requested_experiments`
- `PlotConfig` for `v2.plot_requested_experiments`
- `TrainConfig` for `v2.train`
  Note: by default, `oracle_mode_exp2/exp4` follow `TrainConfig.oracle_mode`, while `oracle_mode_exp3` defaults to `fixed_cipher` so `p_lie` has effect.

Color policy:

- Categorical settings (Experiment 1, Experiment 3, and Experiment 5): `plasma`
- Numerical graph scaling sweeps (Experiment 2 and Experiment 6): `viridis`

## Workflow

### 1) Run experiments (training only) and write manifest

```bash
python3 -m v2.run_requested_experiments \
  --n 5000 \
  --n_values 1000,5000,10000,50000 \
  --sensing S0 \
  --train_steps 100000 \
  --n_env 32 \
  --running_avg_window 500 \
  --fst_k 2 \
  --fst_k_values 1,2,3,4 \
  --d_values 2,4,8,16 \
  --noise_sigmas 0,0.1,0.5,1.0,2.0 \
  --num_seeds 5 \
  --seed 0 \
  --logdir v2/runs \
  --manifest_path plots/requested_experiments_manifest.json
```

This runs all experiment conditions across 5 random seeds and stores run directories in the manifest.

### 2) Plot from manifest (no retraining)

```bash
python3 -m v2.plot_requested_experiments \
  --manifest plots/requested_experiments_manifest.json \
  --plotdir plots
```

Re-run this plotting command anytime to iterate on figures without re-running experiments.

## Outputs

Training runs go to `v2/runs/<run_name>/` with:

- `config.json`
- `metrics.csv`
- `metrics.jsonl`
- `summary.json`
- `spatial_trajectory_with_gradients.png` (PPO hidden + GD/Adam, for `--task spatial`)
- `spatial_trajectory_with_gradients_ppo_comparison.png` (PPO hidden vs visible vs no-oracle)

Learning-curve plots can be regenerated into a run directory with `v2.plot_learning_curves`:

- `success_rate_vs_episodes.png` (for `--task graph`)
- `objective_vs_episodes.png` (for `--task spatial`)
- `distance_vs_episodes.png` (for `--task spatial`)

Plotting outputs go to `plots/`:

- `exp1_encoding_comparison_..._seeds5_ci95.png`
- `exp2_graph_scaling_..._seeds5_ci95.png`
- `exp3_adversarial_lie_rate_..._seeds5_ci95.png`
- `exp4_noisy_sensing_..._seeds5_ci95.png`
- `exp5_fst_k_sweep_..._seeds5_ci95.png`
- `exp6_degree_sweep_..._seeds5_ci95.png`
- `exp6_degree_sweep_path_length_..._seeds5_ci95.png`
- `requested_experiments_summary_seeds5.json`

## Notes

- CPU-only by default; use `--device cuda` (or `cuda:0`) to use GPU when available.
- The plotting script reads existing run outputs from the manifest and does not call training.
