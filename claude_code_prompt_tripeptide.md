# Claude Code Prompt: Tripeptide Conformer Navigation with SDP Oracle

## Context

I have an existing RL codebase (`tasks/spatial/`) that trains PPO agents to optimize non-convex objectives by exploiting hidden convex structure. The current setup uses **synthetic** Fourier-basis lifting maps: the energy is defined as `E(z) = 0.5 * ||F(z) - s*||^2` where `F` is a random Fourier map from R^2 to R^200. The agent receives the hidden-space gradient `F(z) - s*` as an oracle signal. In the synthetic case, `s*` was given by construction.

I want to extend this to a **real physical system** where `s*` is **not known in advance** but is **computed via a semidefinite programming (SDP) relaxation**. This is the key methodological advance: the oracle is no longer assumed — it is constructed from the energy function's Fourier structure alone, without knowledge of the global minimum.

**The molecule** is a tripeptide (Ace-Ala-Ala-Nme) with 4 backbone torsion angles (phi1, psi1, phi2, psi2), giving d=4. Each angle is periodic on [0, 2pi), so the configuration space is a 4D torus T^4. The energy is a Fourier series fitted to an AMBER ff14SB force field evaluation:

```
E(z) = c0 + sum_m [ a_m * cos(m . z) + b_m * sin(m . z) ]
```

where z = (phi1, psi1, phi2, psi2) and m ranges over frequency multi-indices.

**The lifting map** F: T^d -> R^D collects all Fourier basis functions [cos(m.z), sin(m.z)] up to some frequency K_map. Since E is linear in these basis functions, minimizing E on the torus is equivalent to minimizing c^T s over the image of F — which is a non-convex set (it's the image of the torus in R^D).

**The SDP relaxation** replaces the non-convex constraint "s lies on the image of F" with the convex constraint "the trigonometric moment matrix M(s) is positive semidefinite." Solving `min c^T s subject to M(s) >= 0` gives s*_SDP. When the relaxation is tight, s*_SDP = F(z*) exactly, where z* is the true global minimum. The oracle signal is then F(z) - s*_SDP, fed to the RL agent.

**The pipeline:**
1. Load the energy surface (Fourier coefficients from JSON)
2. Construct the lifting map F
3. **Solve the SDP to compute s*_SDP** (no knowledge of z* needed)
4. Use F(z) - s*_SDP as the hidden-gradient oracle
5. Train RL agent with PPO, compare against baselines

The key files in the existing codebase are:

- `spatial_env.py` — `VectorizedSpatialEnv`: the core RL environment. Manages the Fourier map F, computes `_hidden_from_z`, `_gradient_hidden`, `_gradient_xy`, `_objective_value`, handles episode resets, baselines (GD, Adam), and action application. The visible space is `[-coord_limit, coord_limit]^d` with clipping.
- `oracle.py` — `SpatialOracle`: encodes the hidden gradient into a token for the agent. Modes include `convex_gradient` (full hidden gradient), `visible_gradient`, `no_oracle`, etc.
- `model.py` — `PolicyValueNet`: PPO policy/value network. Takes token features (oracle signal + position), normalized objective, step fraction. Outputs continuous action (direction + step scale).
- `ppo.py` — PPO training logic, rollout buffer, GAE.
- `config.py` — `TrainConfig` dataclass with all hyperparameters.
- `train.py` — Main training loop. Builds env, oracle, model, runs PPO training with evaluation and plotting.
- `run_optimizer_studies.py` — Multi-seed experiment runner for meta-optimizer and search-algorithm studies.

## Critical constraint: do not modify other task directories

**Do not touch any file under `tasks/spatial/` or `tasks/alanine_dipeptide/`.** Those directories are existing experiments and must remain unchanged.

Instead, create a **new, self-contained directory `tasks/tripeptide/`** with its own `__init__.py` and all necessary files. If you need functionality from `tasks/spatial/` (e.g., the PPO logic, model architecture, oracle encoding, plotting utilities, training loop structure), **copy the relevant files into `tasks/tripeptide/` and modify the copies.** Do not import from `tasks.spatial` or `tasks.alanine_dipeptide`. The new package should be fully independent.

## Task

Implement a new self-contained package `tasks/tripeptide/` that:
1. Loads a tripeptide energy surface from pre-generated Fourier coefficients
2. Constructs the Fourier lifting map F
3. **Solves an SDP relaxation to compute s*_SDP without knowing the global minimum**
4. Trains an RL agent to navigate the 4D torus using F(z) - s*_SDP as the oracle
5. Compares against visible-gradient, no-oracle, and classical optimizer baselines

## Detailed Implementation Plan

### Step 1: Energy surface

Create `tasks/tripeptide/energy.py` that provides the energy function.

**The energy surface is loaded from a pre-generated JSON file** (`tasks/tripeptide/fourier_coefficients.json`). This file will be generated separately by a one-time OpenMM script and committed to the repo. **You do not need to generate this file — assume it exists.**

The JSON file has this structure:
```json
{
  "d": 4,
  "K_energy": 3,
  "c0": <float>,
  "angle_names": ["phi1", "psi1", "phi2", "psi2"],
  "minima": [
    {"name": "min_0", "angles_rad": [phi1, psi1, phi2, psi2], "energy_kjmol": <float>},
    ...
  ],
  "global_min_index": 0,
  "coefficients": {
    "cos": {"0,0,0,1": <float>, "0,0,1,0": <float>, ...},
    "sin": {"0,0,0,1": <float>, ...}
  }
}
```

The energy is:
```
E(z) = c0 + sum_m a_m * cos(m . z) + b_m * sin(m . z)
```

Provide an `EnergySurface` class with:
- `__init__(json_path)` — load from JSON
- `energy(z: np.ndarray) -> float` — evaluate E at a point on the torus
- `gradient(z: np.ndarray) -> np.ndarray` — analytical gradient dE/dz (d-dimensional)
- `d` — number of torsion angles (4)
- `K_energy` — max frequency
- `global_min` — the known global minimum angles from JSON (used only for evaluation/validation, NOT for oracle construction)
- `c0, cos_coeffs, sin_coeffs` — raw coefficient data accessible for the SDP

**Also provide a synthetic fallback** for testing before the OpenMM JSON is available:
- `generate_synthetic_surface(d=4, K=3, n_minima=8, seed=42) -> EnergySurface`
- Places random Gaussian wells on the torus, fits Fourier coefficients, identifies minima
- Returns an `EnergySurface` with the same interface
- The synthetic surface should be deterministic given a seed

### Step 2: Lifting map F

Create `tasks/tripeptide/lifting_map.py`.

The lifting map F: T^d -> R^D collects Fourier basis functions up to frequency K_map:
```
F(z) = [cos(m . z), sin(m . z)]  for all m with 0 < |m|_inf <= K_map
```

More precisely, enumerate all integer multi-indices m = (m1, ..., md) with each |m_i| <= K_map, excluding m = (0,...,0). For each such m, include both cos(m.z) and sin(m.z). This gives D = 2 * ((2*K_map+1)^d - 1) basis functions.

Provide a `LiftingMap` class with:
- `__init__(d, K_map)` — precompute frequency matrix M_freq of shape (N_freq, d) where N_freq = (2*K_map+1)^d - 1. Store as numpy array for vectorized evaluation.
- `D` — total hidden dimension (= 2 * N_freq)
- `N_freq` — number of distinct frequency multi-indices
- `eval(z: np.ndarray) -> np.ndarray` — evaluate F(z), returns D-dimensional vector. Implementation: `phases = M_freq @ z`, then concatenate `cos(phases)` and `sin(phases)`.
- `jacobian(z: np.ndarray) -> np.ndarray` — returns D x d Jacobian matrix
- `frequency_matrix` — the (N_freq x d) matrix of multi-indices, for use by the SDP
- `energy_as_linear(energy_surface) -> np.ndarray` — returns the D-dimensional vector c such that E(z) ≈ c^T F(z) + c0. This extracts the energy's Fourier coefficients in the ordering used by this lifting map. If K_map >= K_energy, the representation is exact. If K_map < K_energy, it's a truncation.
**Performance note**: at d=4, K_map=2 gives N_freq=624, D=1248. Use vectorized numpy throughout. The frequency matrix should be precomputed once.

### Step 3: SDP relaxation (THIS IS THE KEY NEW COMPONENT)

Create `tasks/tripeptide/sdp_oracle.py`.

This file implements the SDP relaxation that computes s*_SDP — the oracle target — without knowledge of the global minimum z*.

**Background.** The problem is:
```
min_z  E(z) = min_z  c^T F(z) + c0
```
where z is on the torus T^d. The image {F(z) : z in T^d} is non-convex. The SDP relaxation replaces this with:
```
min_s  c^T s   subject to  M(s) >= 0  (positive semidefinite)
```
where M(s) is the **trigonometric moment matrix** — a matrix whose entries are the Fourier basis values, structured so that M(s) >= 0 is a necessary condition for s to lie in the image of F.

**Constructing the moment matrix M(s).**

For the 1D case (single angle theta), the trigonometric moment matrix of order K is a (K+1) x (K+1) Hermitian Toeplitz matrix:
```
M_K = [[c_0,    c_1,    c_2,    ..., c_K   ],
       [c_1*,   c_0,    c_1,    ..., c_{K-1}],
       [c_2*,   c_1*,   c_0,    ..., c_{K-2}],
       ...
       [c_K*,   c_{K-1}*, ...,       c_0    ]]
```
where c_k = E[e^{ik*theta}] are the trigonometric moments, and c_k = (a_k - i*b_k)/2 for our Fourier coefficients (with a_k = cos coefficient, b_k = sin coefficient). The condition M_K >= 0 is equivalent to the sequence (c_0, c_1, ..., c_K) being a valid truncated trigonometric moment sequence (i.e., there exists a positive measure on the circle with these moments).

For the d-dimensional case (torus T^d), we need a **multi-dimensional trigonometric moment matrix**. The approach is:

**Option A (separable relaxation — simpler, weaker):** For each angle i = 1..d independently, form the 1D moment matrix using only the frequency components along that axis. The constraint is M_i >= 0 for all i. This is a product of 1D constraints and is easy to implement but loose.

**Option B (joint moment matrix — tighter, recommended):** Form a single moment matrix indexed by multi-indices. Let I = {m : |m_i| <= K_relax for all i} be the set of multi-indices up to relaxation order K_relax. The moment matrix is indexed by pairs (m, m') in I x I, with entry:
```
M[m, m'] = c_{m - m'}
```
where c_n is the trigonometric moment for multi-index n, expressed in terms of our Fourier basis decision variables:
```
c_n = (a_n - i*b_n) / 2    for n != 0
c_0 = 1                     (normalization of the measure)
c_{-n} = conj(c_n)
```

The matrix M is Hermitian by construction. The constraint M >= 0 encodes that the moments come from a valid probability measure on the torus.

**The SDP problem:**
```
minimize    c_energy^T s
subject to  M(s) >= 0  (PSD constraint)
            s_0 = 1    (normalization: the constant moment is 1)
```

where the decision variables are the moments s = (s_m for all m), and c_energy is the energy coefficient vector aligned to the same indexing. The PSD constraint is linear in s (each entry of M is an affine function of s), so this is a standard SDP.

**Implementation using CVXPY:**
```python
import cvxpy as cp
import numpy as np

def solve_sdp_oracle(lifting_map, energy_surface, K_relax=None):
    """Solve the SDP relaxation to get s*_SDP.
    
    Args:
        lifting_map: LiftingMap instance
        energy_surface: EnergySurface instance  
        K_relax: relaxation order (default: K_map)
    
    Returns:
        s_star: D-dimensional vector in the lifting map's (cos, sin) format
        sdp_bound: lower bound on the energy from the SDP
        status: solver status string
    """
    ...
```

The function should:
1. Build the multi-index set for the moment matrix (all m with |m_i| <= K_relax)
2. Create CVXPY Hermitian matrix variable for M (or real symmetric formulation)
3. Add the PSD constraint M >> 0
4. Add the affine constraints linking M entries to the moment variables s
5. Add normalization constraint c_0 = 1
6. Set the objective: minimize c_energy^T s (energy in terms of moments)
7. Solve (using SCS)
8. Extract s*_SDP from the solution, convert to the real-valued (cos, sin) representation used by the lifting map
9. Return s*_SDP in the same format as `lifting_map.eval(z)` would return

**Handling complex vs real representation.** The moment matrix is naturally Hermitian (complex). The lifting map uses real cos/sin pairs. You need to convert between the two:
```
c_m = (a_m - i*b_m) / 2     (Fourier coeff to complex moment)
a_m = 2*Re(c_m),  b_m = -2*Im(c_m)   (complex moment to Fourier coeff)
```
where a_m is the coefficient of cos(m.z) and b_m is the coefficient of sin(m.z).

Alternatively, you can formulate the SDP entirely in the real domain using a real moment matrix, which avoids complex arithmetic but has twice the dimension. Either approach is fine — use whichever is more natural with CVXPY.

**Important details:**
- K_relax controls the size of the moment matrix: it has (2*K_relax+1)^d rows. For d=4, K_relax=1 gives a 3^4=81 x 81 matrix (feasible). K_relax=2 gives 5^4=625 x 625 (still feasible but slower). Start with K_relax=1.
- The relaxation is tighter at higher K_relax. At sufficiently high K_relax, it becomes exact.
- The SDP is solved ONCE before RL training begins. It's a preprocessing step.
- After solving, verify the solution quality by comparing against the known global minimum from the JSON (for validation only).

**Also provide a validation function:**
```python
def validate_sdp_solution(s_star_sdp, lifting_map, energy_surface):
    """Compare s*_SDP against the known global minimum (for validation only).
    
    This uses energy_surface.global_min which is known from the JSON.
    It is NOT used during oracle construction — only for measuring
    how tight the relaxation is.
    """
    s_star_true = lifting_map.eval(energy_surface.global_min)
    distance = np.linalg.norm(s_star_sdp - s_star_true)
    sdp_energy = float(np.dot(lifting_map.energy_as_linear(energy_surface), s_star_sdp) + energy_surface.c0)
    true_energy = energy_surface.energy(energy_surface.global_min)
    print(f"  ||s*_SDP - F(z*)||: {distance:.6f}")
    print(f"  SDP energy: {sdp_energy:.4f} kJ/mol")
    print(f"  True global min energy: {true_energy:.4f} kJ/mol")
    print(f"  Gap: {sdp_energy - true_energy:.4f} kJ/mol")
    return {"distance": distance, "sdp_energy": sdp_energy, "true_energy": true_energy}
```

### Step 4: Environment class

Create `tasks/tripeptide/env.py` with class `VectorizedTripeptideEnv` that:

1. **Mirrors the interface of `VectorizedSpatialEnv`** (same `get_obs`, `step`, `reset` signatures, same observation dict keys, same info dict keys).

2. **Visible space is the d-dimensional torus [0, 2pi)^d.**
   - `_apply_action` wraps coordinates modulo 2pi instead of clipping
   - `_apply_baseline_optimizer_step` also wraps modulo 2pi
   - `_sample_xy` samples uniformly from [0, 2pi)^d
   - Position normalization for the policy: `z / pi - 1`, so [0, 2pi) maps to [-1, 1)

3. **Takes s*_SDP as a constructor argument.** The environment does not solve the SDP — it receives s*_SDP from the training script and stores it. This is the oracle target for all episodes.

4. **Fixed landscape, random starts.** Each episode samples a random starting point on the torus. The energy landscape and oracle target are the same every episode. The agent may learn to memorize optimal paths — this is fine, because the oracle was constructed without knowledge of z*, so there is no circularity.

5. **Hidden gradient oracle**: `_gradient_hidden` returns `lifting_map.eval(z) - s_star_sdp`.

6. **Energy evaluation**: `energy_surface.energy(z)`.

7. **Visible gradient**: `energy_surface.gradient(z)` for the visible-gradient oracle and for GD/Adam baselines.

8. **Objective function**: `0.5 * ||F(z) - s*_SDP||^2`. This is convex in hidden space and has the SDP oracle target as its minimum.

9. **Action space**: direction (d-dimensional) + step scale (1 scalar) = (d+1)-dimensional continuous action.

10. **Baselines (GD, Adam)** operate on the torus with wrapping, using the visible-space energy gradient. They follow the energy downhill. They don't know about the oracle. They'll reach whatever local minimum is nearest.

11. **Random search baseline**: sample random points on the torus, keep the one with the lowest objective (Fourier-space distance to s*_SDP).

### Step 5: Config and training integration

Create `tasks/tripeptide/config.py` by copying `tasks/spatial/config.py` and modifying:
- Remove synthetic-specific fields
- Add:
  - `K_map: int = 2` — lifting map frequency (controls D)
  - `K_relax: int = 1` — SDP relaxation order
  - `energy_json: str = "tasks/tripeptide/fourier_coefficients.json"` — path to coefficients
  - `use_synthetic_fallback: bool = False` — use synthetic energy if JSON missing
  - `synthetic_d: int = 4` — dimension for synthetic fallback
  - `synthetic_K: int = 3` — Fourier order for synthetic fallback
  - `synthetic_n_minima: int = 8` — number of basins for synthetic fallback
- Defaults: `visible_dim = 4`, `max_horizon = 120`, `train_steps = 500000`

Create `tasks/tripeptide/train.py` by copying `tasks/spatial/train.py` and modifying:
- Import from `tasks.tripeptide`
- **Before building the env**, solve the SDP:
  ```python
  energy_surface = load_energy_surface(config.energy_json)
  lifting_map = LiftingMap(d=energy_surface.d, K_map=config.K_map)
  s_star_sdp, sdp_bound, status = solve_sdp_oracle(lifting_map, energy_surface, K_relax=config.K_relax)
  print(f"SDP solved: bound={sdp_bound:.4f}, status={status}")
  validate_sdp_solution(s_star_sdp, lifting_map, energy_surface)
  ```
- Pass s_star_sdp to the env constructor
- Everything else (oracle, model, PPO, training loop) stays the same

Copy over and adapt:
- `tasks/tripeptide/model.py` — copy from spatial, no changes needed (dimension-agnostic)
- `tasks/tripeptide/oracle.py` — copy from spatial, no changes needed
- `tasks/tripeptide/ppo.py` — copy from spatial, no changes needed
- `tasks/tripeptide/plotting.py` — copy from spatial, adapt: no 2D heatmaps, plot objective vs step curves, final objective bar chart, optionally pairwise angle projections
- `tasks/tripeptide/__init__.py` — empty or minimal

### Step 6: Evaluation and experiments

Create `tasks/tripeptide/run_experiment.py` that:

**Experiment 1: Oracle comparison** (primary result)
- Compare: hidden-gradient PPO (SDP oracle), visible-gradient PPO, no-oracle PPO, Adam, GD, random search
- Multiple seeds
- This is the main result: does the SDP-constructed oracle help the RL agent find the global minimum?

**Experiment 2: SDP tightness** (diagnostic)
- Sweep K_relax (1, 2, maybe 3 if feasible) and measure:
  - SDP bound vs true global minimum energy
  - ||s*_SDP - F(z*)|| (distance in Fourier space)
  - RL success rate with each s*_SDP
- This shows whether the relaxation is tight enough to be useful

**Experiment 3: K_map sweep** (ablation)
- Sweep K_map (1, 2, 3) at fixed K_relax
- Shows how oracle dimensionality affects RL performance
- At K_map=1, D=160; at K_map=2, D=1248; at K_map=3, D=4800

## Key constraints

- **Do not modify anything under `tasks/spatial/` or `tasks/alanine_dipeptide/`.** Copy files into `tasks/tripeptide/` if needed.
- **`tasks/tripeptide/` must be fully self-contained.** No imports from other task packages.
- **CVXPY is the only new dependency** (for the SDP). Add it to requirements.txt. Use SCS as the solver (free, handles SDPs). Don't require MOSEK.
- **The SDP is solved once before training.** It is NOT in the training loop. It's a preprocessing step.
- **s*_SDP is computed without knowledge of z*.** The `energy_surface.global_min` field is used ONLY for validation (measuring how tight the relaxation is), never for oracle construction.
- **Keep the env interface identical** to `VectorizedSpatialEnv` patterns.
- **Performance matters at D=1248.** Use vectorized numpy for lifting map eval/Jacobian. The SDP itself can be slow (it runs once) but the per-step operations must be fast.

## Expected output

```bash
# With real energy coefficients
python -m tasks.tripeptide.train \
    --K_map 2 \
    --K_relax 1 \
    --energy_json tasks/tripeptide/fourier_coefficients.json \
    --oracle_mode convex_gradient \
    --sensing S0 \
    --train_steps 500000 \
    --max_horizon 120 \
    --seed 0

# With synthetic fallback (for testing)
python -m tasks.tripeptide.train \
    --use_synthetic_fallback \
    --K_map 2 \
    --K_relax 1 \
    --oracle_mode convex_gradient \
    --sensing S0 \
    --train_steps 300000 \
    --seed 0
```

The training script should print at startup:
```
Loading energy surface from tasks/tripeptide/fourier_coefficients.json
  d=4, K_energy=3, 12 known minima
Constructing lifting map: K_map=2, D=1248
Solving SDP relaxation (K_relax=1, moment matrix size=81x81)...
  SDP status: optimal
  SDP energy bound: -42.31 kJ/mol
  True global minimum energy: -43.17 kJ/mol  (gap: 0.86 kJ/mol)
  ||s*_SDP - F(z*)||: 0.0234  (tightness check)
Constructing environment with SDP oracle...
Training PPO agent...
```

## Files to create (all under `tasks/tripeptide/`)
1. `__init__.py`
2. `energy.py` — EnergySurface class, JSON loading, synthetic fallback, gradient computation
3. `lifting_map.py` — LiftingMap class, eval, Jacobian, frequency index management
4. `sdp_oracle.py` — SDP relaxation construction and solution, validation, moment matrix construction
5. `env.py` — `VectorizedTripeptideEnv` class
6. `config.py` — copied from spatial, adapted for tripeptide
7. `model.py` — copied from spatial (unchanged)
8. `oracle.py` — copied from spatial (unchanged)
9. `ppo.py` — copied from spatial (unchanged)
10. `train.py` — copied from spatial, adapted: loads energy, solves SDP, builds env
11. `plotting.py` — copied from spatial, adapted for d=4
12. `run_experiment.py` — multi-seed evaluation with oracle comparison, K_relax sweep, K_map sweep
13. `requirements.txt` — adds cvxpy, scs to existing deps

## Files to NOT modify
- Everything under `tasks/spatial/`
- Everything under `tasks/alanine_dipeptide/`
