# Optimizer Experiments Report

Date: 2026-03-14

## Configuration

This report covers the full optimizer study suite run with the following settings:

| Parameter | Value |
|---|---|
| Visible dimension | 3 |
| Hidden dimension | 300 |
| Coordinate limit | 1 |
| Map type | Fourier |
| Basis complexity | 3 |
| Policy architecture | MLP (128 hidden) |
| Training steps | 150,000 |
| Evaluation horizon | 60 |
| Meta tasks per seed | 500 |
| Search max budget | 300,000 env steps |
| Seeds | 10 (0--9) |
| Success curriculum | Disabled (both studies) |
| Search RL mode | Deterministic |

Compared to earlier runs (2D, D=100, 3 seeds), this configuration is substantially harder: the visible space is 3D, the hidden space is 3x larger (300), the coordinate range is tighter ([-1,1]^3), and the Fourier landscape is correspondingly denser. The policy is smaller (128 vs 256 hidden units) and trains for half as many steps.

## Study 1: Meta-Optimizer

The meta-optimizer study evaluates each method on 500 fresh (start, target) pairs per seed, using a single 60-step rollout. The question: given one attempt on a new task, which method gets closest to the optimum?

### Final Objective

| Method | Final objective | 95% CI |
|---|---:|---:|
| Random search | **0.148** | 0.006 |
| RL hidden gradient | **0.164** | 0.086 |
| Adam | 0.317 | 0.017 |
| GD | 0.323 | 0.009 |
| RL visible oracle | 0.383 | 0.016 |
| RL no oracle | 0.637 | 0.022 |

### Best Objective Visited

| Method | Best objective | 95% CI |
|---|---:|---:|
| RL hidden gradient | **0.096** | 0.062 |
| Random search | 0.148 | 0.006 |
| RL visible oracle | 0.270 | 0.014 |
| GD | 0.323 | 0.009 |
| Adam | 0.317 | 0.010 |
| RL no oracle | 0.444 | 0.051 |

### Interpretation

**RL hidden gradient and random search are the top two methods, well ahead of the rest.** On final objective, random search edges ahead (0.148 vs 0.164), but the confidence intervals overlap. On best-visited objective, the hidden-gradient policy is clearly the strongest at 0.096, indicating it passes through better states than any other method but gives some back by the end of the rollout.

**The gap between best-visited and final is the policy's main weakness.** The hidden-gradient policy finds states at 0.096 on average but finishes at 0.164 -- a 70% overshoot. This gap is much larger than in the earlier 2D/D=100 runs, suggesting that the harder 3D/D=300 landscape is more punishing when the policy takes steps that are too aggressive near good regions.

**GD and Adam are stuck at ~0.32.** Both gradient-based baselines converge to the same plateau. The 3D Fourier landscape has enough local structure that local descent in the visible space gets trapped well before reaching the global minimum. This is worse than the 2D case, consistent with the higher-dimensional visible space creating more complex local minima.

**RL visible oracle is worse than GD/Adam.** This is notable: giving an RL policy the true visible-space gradient actually produces worse results than just running gradient descent with that same gradient. The policy trained on the visible gradient has not learned anything useful beyond what direct descent already provides, and the RL overhead (stochastic training, finite capacity) makes it worse.

**RL no oracle is the worst method by far at 0.637.** Without the oracle signal, the policy has learned essentially nothing useful. This is the control that confirms the hidden gradient is carrying real information.

### Per-Seed Variance

The hidden-gradient policy shows high variance across seeds:

| Seed | Final | Best |
|---|---:|---:|
| 0 | 0.092 | 0.044 |
| 1 | 0.165 | 0.073 |
| 2 | 0.096 | 0.052 |
| 3 | 0.208 | 0.110 |
| 4 | 0.192 | 0.123 |
| 5 | 0.125 | 0.065 |
| 6 | **0.035** | **0.013** |
| 7 | 0.039 | 0.014 |
| 8 | 0.548 | 0.378 |
| 9 | 0.145 | 0.093 |

Seed 8 is a clear outlier (0.548 final), pulling the mean up substantially. Seven of ten seeds finish below 0.2, and the best seeds (6, 7) reach 0.035--0.039, which is far better than any other method. The variance reflects genuine difficulty differences across randomly sampled Fourier maps, not just optimization noise.

### Optimization Curves

The optimization-step curve (meta_optimizer_objective_vs_step.png) shows:

- **RL hidden gradient** (yellow) drops rapidly in the first 5--8 steps and continues declining throughout the horizon. It reaches ~0.15 by step 60.
- **Random search** (magenta) declines steadily across the horizon as more random samples are drawn.
- **GD and Adam** (blue, purple) drop quickly in the first 10 steps but flatten around 0.32.
- **RL visible oracle** (orange) descends more slowly and plateaus around 0.38.
- **RL no oracle** (pink) barely moves, staying above 0.6.

The hidden-gradient policy has a distinctive two-phase trajectory: rapid descent (steps 0--8) followed by slow continued improvement. This suggests it learns a coarse approach strategy first and a finer refinement strategy second.

## Study 2: Search Algorithm

The search study evaluates each method as a repeated-episode search algorithm on one fixed task per seed, under an environment-step budget up to 300,000. RL policies are run deterministically. The question: given many attempts with a fixed step budget, which method finds the best solution?

### Budget Curve (Selected Budgets)

| Budget | Random search | RL no oracle | RL visible oracle | RL hidden gradient |
|---:|---:|---:|---:|---:|
| 62 | **0.167** | 0.402 | 0.355 | 0.395 |
| 203 | **0.079** | 0.211 | 0.217 | 0.198 |
| 486 | **0.039** | 0.067 | 0.099 | 0.128 |
| 993 | **0.023** | 0.036 | 0.031 | 0.088 |
| 2,027 | **0.019** | 0.035 | **0.015** | 0.057 |
| 4,851 | 0.009 | 0.009 | **0.008** | 0.017 |
| 9,906 | 0.007 | 0.008 | **0.006** | 0.013 |
| 300,000 | 0.006 | 0.007 | **0.005** | **0.004** |

### Final (300k Budget) Summary

| Method | Best objective | 95% CI | Wall time | Episodes used |
|---|---:|---:|---:|---:|
| RL hidden gradient | **0.0044** | 0.0026 | 11.2 s | 225 |
| RL visible oracle | 0.0049 | 0.0025 | 2.4 s | 52 |
| Random search | 0.0063 | 0.0023 | 0.08 s | 94 |
| RL no oracle | 0.0068 | 0.0019 | 3.1 s | 65 |

### First Success (objective <= threshold)

| Method | Mean steps | Median | Min | Max |
|---|---:|---:|---:|---:|
| RL visible oracle | **3,904** | 2,656 | 320 | 10,432 |
| RL no oracle | 4,397 | 3,952 | 608 | 12,128 |
| Random search | 5,584 | 4,844 | 1,331 | 14,576 |
| RL hidden gradient | 14,256 | 4,496 | 1,664 | 72,096 |

### Interpretation

The search results tell a different story from the meta-optimizer. There are three regimes:

**Low budget (<1,000 steps): random search dominates.** At 62 steps, random search is already at 0.167 while all RL methods are above 0.35. Random search has the advantage of zero overhead: each sample is one environment step, while RL methods require a full episode (60 steps) to produce one candidate. At small budgets, random search has simply drawn more candidates.

**Mid budget (1,000--10,000 steps): RL visible oracle emerges.** Starting around 2,000 steps, the RL visible oracle takes the lead and holds it through 10,000 steps. This is a reversal from the meta study, where the visible oracle was one of the weakest methods. The explanation is that deterministic search with the visible-oracle policy produces consistent, high-quality trajectories when run repeatedly, even though a single rollout is mediocre. The visible gradient signal is locally correct, and repeated restarts from different initial conditions allow the policy to cover the landscape efficiently.

**High budget (>100,000 steps): RL hidden gradient catches up and wins.** At the maximum 300k budget, the hidden-gradient policy achieves the best objective (0.0044), narrowly beating the visible oracle (0.0049). But it took 225 episodes and 11 seconds to get there, compared to 52 episodes and 2.4 seconds for the visible oracle.

**The hidden-gradient policy is the slowest search algorithm.** Its first-success mean is 14,256 steps, roughly 3x slower than the RL visible oracle (3,904) and random search (5,584). The median is more reasonable at 4,496, but the max is 72,096 -- one seed where the policy struggled extensively. This high variance is consistent with the meta-optimizer findings: the policy is strong when the Fourier map is favorable, but fragile when it is not.

**Random search saturates early.** After about 50,000 steps (94 episodes), random search stops improving, plateauing at 0.006. It cannot refine below that level because it samples uniformly over [-1,1]^3 and the probability of landing very close to the optimum is vanishingly small. The RL methods continue improving beyond this point.

## Combined Interpretation

### The Meta/Search Split

The central finding, consistent with the earlier 2D results, is:

- **RL hidden gradient is the strongest single-rollout optimizer** (meta study: best-visited = 0.096).
- **RL hidden gradient is the slowest search algorithm** at low-to-mid budgets.
- **RL visible oracle, mediocre in meta, is the strongest mid-budget search method.**
- **Random search remains the most efficient low-budget search method.**

This split is sharper than in the 2D case. In 2D, the hidden-gradient policy won both the meta study and the high-budget search. In 3D/D=300, it still wins the meta study (on best-visited) and the max-budget search, but the margin is smaller and the variance is much higher.

### What Changed from 2D to 3D

| Aspect | 2D / D=100 | 3D / D=300 |
|---|---|---|
| Hidden gradient meta (final) | 0.014 | 0.164 |
| GD/Adam meta (final) | ~0.20 | ~0.32 |
| Random search meta (final) | 0.035 | 0.148 |
| Hidden gradient dominance (meta) | Clear winner | Tied with random search |
| Hidden gradient search (300k) | Worst | Best |
| Per-seed variance (hidden gradient) | Low | High |

The harder setting (3D visible, D=300, coord_limit=1, 150k train steps) has compressed the gap between the hidden-gradient policy and simpler methods. The policy still extracts real value from the oracle signal -- it is 2x better than GD/Adam and its best-visited states are far better than anything else -- but the final-objective advantage over random search has narrowed to the point of statistical overlap.

The most likely explanations are:
1. **Undertraining.** 150k steps may not be enough for a 128-unit MLP to learn the 3D-to-300D Fourier relationship. The 2D results used 300k steps with a 256-unit network.
2. **Harder landscape.** The [-1,1]^3 domain with D=300 and basis_complexity=3 creates a much denser, rougher landscape. The policy must learn a 3-to-300 mapping instead of 2-to-100, with tighter coordinate constraints.
3. **Overshoot.** The policy's best-visited (0.096) is far better than its final (0.164), suggesting it approaches good regions but cannot reliably stop there.

### What the Visible Oracle Reveals

The visible oracle results are newly informative in 3D. In the meta study, the visible oracle is worse than GD/Adam (0.383 vs 0.323). But in search, it is the best method for budgets 2,000--50,000. This pattern suggests that the visible-gradient RL policy has learned a decent deterministic controller -- not as good as raw gradient descent on a single pass, but useful for generating diverse trajectories across random restarts. The meta result is worse because a single stochastic rollout from the visible-oracle policy is noisy, but deterministic search with restarts averages out that noise.

## What To Do Next

1. **Increase training budget.** Run the 3D/D=300 configuration with 300k--500k training steps and a 256-unit policy to match the 2D compute budget.
2. **Add stopping/refinement.** The 70% overshoot gap (best=0.096, final=0.164) is the single largest source of lost performance. A step-size schedule or learned stopping criterion would directly address this.
3. **Hybrid search.** Use the hidden-gradient policy for the first 1--2 rollouts (it finds good regions fast), then switch to random search or local refinement for the remaining budget.
4. **Visible-dimension sweep.** The 2D-to-3D comparison suggests performance degrades with visible dimension. A systematic sweep (2, 3, 4, 5) would quantify this.
5. **Deterministic meta evaluation.** The meta study uses stochastic rollouts. Running it deterministically (as the search study does) would isolate whether the variance is from the policy's stochasticity or from the landscape.

## Appendix: Run Command

```bash
python3 -m v2.run_spatial_optimizer_studies \
  --study all \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --suite_output_dir plots/spatial_optimizer_studies \
  --logdir v2/runs \
  --spatial_coord_limit 1 \
  --spatial_visible_dim 3 \
  --spatial_hidden_dim 300 \
  --train_steps 150000 \
  --search_disable_success_curriculum \
  --meta_disable_success_curriculum \
  --policy_hidden_dim 128 \
  --spatial_policy_arch mlp \
  --search_rl_deterministic
```
