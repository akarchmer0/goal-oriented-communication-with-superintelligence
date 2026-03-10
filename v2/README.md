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
  --spatial_step_size 0.35 \
  --spatial_sgd_gradient_noise_std 0.1 \
  --spatial_success_threshold 1.0 \
  --sensing S0 \
  --train_steps 300000
```

Regenerate the spatial learning curves afterward without retraining:

```bash
python3 -m v2.plot_learning_curves \
  --run_dir v2/runs/<run_name>
```

Interpretation:
- hidden objective is convex in `R^D`: `E(s)=0.5||s-s*||^2`
- human controls only `z ∈ R^2`; hidden state is `s=F(z)` with nonlinear sinusoidal basis
- oracle sends the true hidden-space gradient `g_t = s_t - s*` each step
- policy learns `(g_t, z_t) -> a_t` with **continuous 2D actions** (any direction), using reward from energy decrease `E(F(z_t)) - E(F(z_{t+1}))`
- spatial diagnostics now include both `2D GD` and noisy-gradient `2D SGD` baselines

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
- `spatial_trajectory_with_gradients.png` (for `--task spatial`)

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

- CPU-only (no GPU required or used).
- The plotting script reads existing run outputs from the manifest and does not call training.
