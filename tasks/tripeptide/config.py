from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    task: str = "tripeptide"
    visible_dim: int = 4
    K_map: int = 2
    K_relax: int = 1
    energy_json: str = "tasks/tripeptide/fourier_coefficients.json"
    use_synthetic_fallback: bool = False
    synthetic_d: int = 4
    synthetic_K: int = 3
    synthetic_n_minima: int = 8
    token_noise_std: float = 0.0
    ppo_step_scale: float = 1.0
    step_size: float = 0.3
    success_threshold: float = 0.01
    enable_baselines: bool = True
    tune_baseline_lrs: bool = True
    baseline_lr_candidates: str = "0.001,0.003,0.01,0.03,0.05,0.1,0.2"
    baseline_lr_tune_tasks: int = 64
    optimization_curve_tasks: int = 100
    enable_optimization_curve_eval: bool = True
    policy_arch: str = "mlp"
    plot_interval_episodes: int = 100
    enable_objective_plateau_early_stop: bool = False
    objective_plateau_patience_episodes: int = 1000
    objective_plateau_min_delta: float = 1e-3
    objective_plateau_warmup_episodes: int = 500
    n_env: int = 32
    algo: str = "ppo"
    train_steps: int = 500_000
    rollout_len: int = 64
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    lr_scheduler: str = "cosine"
    lr_min_factor: float = 0.1
    lr_warmup_updates: int = 0
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    oracle_proj_dim: int = 64
    token_embed_dim: int = 16
    sensing: str = "S0"
    reward_noise_std: float = 0.0
    oracle_mode: str = "convex_gradient"
    max_horizon: int = 120
    s1_step_penalty: float = -0.01
    running_avg_window: int = 100
    save_metrics_interval_episodes: int = 500
    eval_interval_episodes: int = 200
    eval_episodes: int = 200
    minibatches: int = 4
    seed: int = 0
    logdir: str = "runs"
    run_name: str = ""
    device: str = "cpu"
    enable_training_plots: bool = True

    @property
    def spatial_hidden_dim(self) -> int:
        """Hidden dim = D of the lifting map."""
        from .lifting_map import LiftingMap
        lm = LiftingMap(d=self.visible_dim, K_map=self.K_map)
        return lm.D

    @property
    def spatial_token_dim(self) -> int:
        """Token dim matches hidden dim for convex_gradient mode."""
        if self.oracle_mode == "convex_gradient":
            return self.spatial_hidden_dim
        return self.spatial_hidden_dim

    def resolve_run_dir(self) -> Path:
        base = Path(self.logdir)
        default_run_name = (
            f"tripeptide_Kmap{self.K_map}_Krelax{self.K_relax}_"
            f"{self.oracle_mode}_{self.sensing}_seed{self.seed}"
        )
        run_name = self.run_name or default_run_name
        return base / run_name


@dataclass(frozen=True)
class PlotConfig:
    plotdir: str = "plots"
    max_points: int = 2500
