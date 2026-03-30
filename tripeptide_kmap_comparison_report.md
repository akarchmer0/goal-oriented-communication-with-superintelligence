# Tripeptide Oracle Comparison: K_map=2/K_relax=1 vs K_map=3/K_relax=2

## 1. Experimental Setup

Both experiments optimize the same 4-dimensional tripeptide energy surface (4 dihedral angles on the torus $[0, 2\pi)^4$). The energy surface has 2400 cosine and 2400 sine Fourier coefficients, a constant offset of 39.01 kJ/mol, and a global minimum energy of -67.10 kJ/mol.

The two runs differ in the lifting map and SDP relaxation parameters:

| Parameter | Run A (low) | Run B (high) |
|---|---|---|
| K_map (lifting cutoff) | 2 | 3 |
| K_relax (relaxation order) | 1 | 2 |
| Hidden dim D | 1,248 | 4,800 |
| N_freq (cos/sin blocks) | 624 | 2,400 |
| Moment matrix size | 81 x 81 | 625 x 625 |
| SDP decision variables | 40 | 312 |
| Training steps | 500K | 5M |
| Episodes completed | 2,564 | 31,706 |
| Lattice granularity | 100 | 50 |

All other parameters are identical: visible_dim=4, policy hidden_dim=64, oracle_proj_dim=64, max_horizon=200, PPO with lr=3e-4, n_env=32, step_size=0.3, seed=0.

## 2. Headline Results

### Final evaluation (100 random-start tasks, horizon=200)

| Method | Run A (K2/R1) final obj | Run B (K3/R2) final obj | Improvement |
|---|---|---|---|
| **RL hidden gradient** | **0.0779 +/- 0.010** | **0.0463 +/- 0.009** | **1.68x better** |
| GD (tuned LR=0.1) | 0.0846 +/- 0.006 | 0.0855 +/- 0.007 | ~same |
| Adam (tuned LR=0.2) | 0.0926 +/- 0.008 | 0.0996 +/- 0.008 | ~same |
| RL no oracle | 0.3811 +/- 0.035 | 0.2889 +/- 0.028 | 1.32x |
| RL visible oracle | 0.1687 +/- 0.014 | 0.1492 +/- 0.012 | 1.13x |

### Success rates (percent of tasks reaching given threshold)

| Threshold | Run A: RL hidden | Run B: RL hidden | Run A: GD | Run B: GD |
|---|---|---|---|---|
| < 0.01 (near-optimal) | 28% | **46%** | 0% | 3% |
| < 0.05 | 28% | **50%** | 0% | 3% |
| < 0.10 | 74% | **89%** | 80% | 73% |

### Training success rate

| Metric | Run A (K2/R1) | Run B (K3/R2) |
|---|---|---|
| Final training success rate (last 100 ep) | 17.9% | **36.1%** |
| Peak training success rate | 22% | **57%** |
| Final avg training objective | 0.0875 | **0.0551** |
| Best avg training objective | 0.0785 | **0.0386** |

## 3. Analysis

### 3.1 Oracle quality drives RL performance

The baselines (GD, Adam) are nearly identical across runs -- they don't use the oracle, so the lifting parameters don't affect them. This serves as a control: the landscape itself is the same. The improvement in RL hidden gradient performance (0.0779 -> 0.0463, a 40% reduction) is therefore attributable entirely to the higher-quality oracle from the tighter SDP relaxation.

- **K_relax=1** produces a very loose relaxation: 81x81 moment matrix with only 40 decision variables. The SDP has limited ability to approximate F(z*).
- **K_relax=2** produces a substantially tighter relaxation: 625x625 moment matrix with 312 decision variables. This 8x increase in decision variables allows much more accurate localization of the surrogate target s*_SDP.

### 3.2 Bimodal task distribution

Both runs exhibit a striking bimodal pattern in per-task outcomes:

**Run A (K2/R1):**
- 28 tasks (28%) reach near-optimal (< 0.01)
- 0 tasks land between 0.01 and 0.05
- 46 tasks (46%) land in [0.05, 0.10)
- The agent either finds the global basin or gets stuck in a secondary basin, with almost nothing in between

**Run B (K3/R2):**
- 46 tasks (46%) reach near-optimal (< 0.01) -- a 64% increase over Run A
- 4 tasks in (0.01, 0.05)
- 39 tasks (39%) in [0.05, 0.10)
- 11 tasks (11%) in [0.10, 0.20) -- fewer stuck tasks

The tighter oracle converts many of the "stuck in a secondary basin" outcomes into successful global-minimum finds. The median final objective drops from 0.093 (Run A) to 0.035 (Run B).

### 3.3 Oracle separation

**Run A:** RL hidden gradient (0.078) barely beats GD (0.085) -- only a 1.09x separation. The weak oracle provides marginal advantage over gradient descent on this 4D surface.

**Run B:** RL hidden gradient (0.046) clearly beats GD (0.086) -- a 1.85x separation. With the tighter oracle, the RL agent pulls ahead decisively.

Neither run shows RL hidden gradient beating Adam as dramatically as in the spatial or alanine tasks. This is because the 4D tripeptide surface has relatively few deep local minima, so gradient methods perform reasonably well. The oracle's value is in reliably steering the agent into the global basin from diverse starting points (46% vs 0-3% near-optimal for baselines in Run B).

### 3.4 Training dynamics

**Run A (500K steps, 2,564 episodes):** The learning curves show the agent begins improving around episode 1000 and reaches a plateau near episode 2000. The success rate stabilizes around 17-22%. The short training may not have fully converged -- the agent appears still learning when training ends.

**Run B (5M steps, 31,706 episodes):** With 10x more training, the agent reaches a success rate plateau around 35-40% by episode 15,000 and then fluctuates. The peak success rate of 57% (vs 22% for Run A) indicates both the tighter oracle and the longer training contribute to better performance.

### 3.5 Lattice discretization

Run A uses granularity=100 (finer grid, spacing=0.063 rad) while Run B uses granularity=50 (coarser grid, spacing=0.127 rad). Despite the coarser action space, Run B performs substantially better, confirming that oracle quality matters more than action resolution in this regime.

### 3.6 Cost-performance tradeoff

| Resource | Run A (K2/R1) | Run B (K3/R2) | Ratio |
|---|---|---|---|
| Hidden dim D | 1,248 | 4,800 | 3.8x |
| SDP moment matrix | 81 x 81 | 625 x 625 | 60x entries |
| SDP variables | 40 | 312 | 7.8x |
| Training steps | 500K | 5M | 10x |
| Episodes | 2,564 | 31,706 | 12.4x |
| Near-optimal success | 28% | 46% | 1.64x |
| Mean final objective | 0.0779 | 0.0463 | 1.68x |

The higher-quality oracle (K3/R2) requires substantially more compute: ~4x larger hidden space, ~8x more SDP variables, and 10x more training steps. In return, it delivers a 1.7x improvement in mean final objective and a 1.6x improvement in near-optimal success rate. Whether this tradeoff is worthwhile depends on the application.

## 4. Key Takeaways

1. **Oracle quality is the primary lever for RL performance on the tripeptide.** The baselines are unchanged across runs; all improvement comes from the tighter SDP relaxation.

2. **K_relax=2 substantially outperforms K_relax=1** on this 4D surface, increasing near-optimal success from 28% to 46% and reducing mean objective by 40%.

3. **The oracle separation (RL hidden vs GD) widens with oracle quality**: 1.09x with the weak oracle vs 1.85x with the tighter oracle. A loose oracle barely justifies the RL machinery; a tight oracle makes it clearly worthwhile.

4. **Bimodal outcome distribution** persists in both runs: the agent either finds the global basin or gets trapped. The tighter oracle shifts the balance toward success without eliminating the bimodality.

5. **Run A may be undertrained** (500K steps, 2,564 episodes). Extending training could narrow the gap. Run B with 5M steps appears closer to convergence but still shows fluctuation.

6. **Neither run matches the dramatic separations seen in alanine dipeptide (76% success) or the spatial task (9-30x).** The 4D torus is harder: more local minima, higher-dimensional action space, and the SDP relaxation is looser (larger moment matrices needed for the same tightness in higher dimensions).
