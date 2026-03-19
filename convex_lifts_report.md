# Learning to Optimize through Convex Lifts via Reinforcement Learning
*Empirical evidence that RL agents can exploit hidden convex structure in non-convex optimization landscapes*

---

## 1. Motivation: The Structure of Protein Conformational Search

Many important problems in the sciences share a common structure: they are naturally posed in a small number of interpretable dimensions with physical meaning, yet are fundamentally non-convex in that space. Standard gradient-based optimization is therefore susceptible to local minima, saddle points, and initialization sensitivity.

The canonical example is protein conformational search. A protein's 3D structure is determined by the backbone torsion (dihedral) angles — the angles of rotation around each bond in the chain. These angles $(\phi, \psi)$ are periodic on $[0, 2\pi)$, so the natural configuration space is a torus. Even for the simplest model system — alanine dipeptide, with just two torsion angles — the free energy landscape on this 2D torus exhibits multiple local minima (the $C_{7\text{eq}}$, $C_{7\text{ax}}$, and $\alpha_R$ conformers), separated by energy barriers of several kcal/mol. For a real protein with $N$ residues, the landscape lives on an approximately $2N$-dimensional torus (plus side-chain torsion angles), and the number of local minima grows combinatorially.

Crucially, this is not a problem that needs to be solved once. Every copy of a protein, synthesized as a linear chain by the ribosome, begins in a different random unfolded configuration and must navigate to the same native state (the global free energy minimum) in real time. The landscape is fixed — determined by the amino acid sequence — but every individual molecule must reliably reach the global minimum from an arbitrary starting point. This is Levinthal's paradox: exhaustive search of the torsion space would take longer than the age of the universe, yet proteins fold within milliseconds to seconds. The resolution is that the landscape is "funneled" — there is a global bias toward the native state.

A key question arises: if the conformational energy landscape is non-convex on the torus but admits a convex reformulation in a higher-dimensional space, can an optimizer *learn* to exploit that hidden convex structure — effectively learning the funnel — without explicit knowledge of the lift?

---

## 2. Non-Convex Problems with Known Convex Lifts

The torsion-angle conformational search problem belongs to a well-studied class: problems that are non-convex in their natural low-dimensional space but admit exact or near-exact convex reformulations in a higher-dimensional lifted space.

The configuration space of torsion angles is a product of circles — the torus $\mathbb{T}^n = (S^1)^n$. Points on the circle $S^1$ can be represented as unit complex numbers, and the set of rank-1 positive semidefinite Hermitian matrices of the form $xx^*$ (where $x$ lies on the unit circle) can be relaxed to the full PSD cone. This is precisely the mechanism behind the Kuramoto synchronization relaxation: oscillator phases on the torus are lifted into Hermitian PSD matrices, where the coupling objective becomes convex. The same algebraic structure applies to torsion-angle optimization — each periodic angle can be lifted into a PSD matrix variable, and trigonometric coupling terms in the energy become linear in the lifted variables.

More concretely, molecular force fields express torsional energy contributions as Fourier series in the dihedral angles: $V(\phi) = \sum_n V_n [1 + \cos(n\phi - \gamma_n)]$. These are trigonometric polynomials — exactly the class of functions for which PSD lifts via the trigonometric moment matrix are known to produce tight convex relaxations (the Lasserre/sum-of-squares hierarchy on the torus). The non-convexity arises from the periodicity and the interaction between multiple torsion angles; the lift into the PSD cone resolves both.

The exactness of such lifts holds under conditions (low noise, sufficient measurements, special graph structure in the coupling). In the protein setting, the relevant question is whether the energy landscape of real force fields (AMBER, CHARMM, OPLS) — which include van der Waals, electrostatic, and solvent terms beyond the pure torsional Fourier series — remains approximately liftable. This motivates learning the lift from data rather than deriving it analytically.

---

## 3. The Invertibility Bottleneck in Learned Lifts

A natural extension is to learn the lifting map from data rather than deriving it analytically. The existing literature has pieces of this: Input Convex Neural Networks (ICNNs, Amos et al. 2017) construct networks whose output is provably convex in their input; normalizing flows learn invertible maps; Koopman operator methods learn liftings under which nonlinear dynamics become linear.

However, all practically useful learned lifts have required invertibility (or at minimum injectivity) to be useful. The reason is straightforward: to apply an optimizer in z-space and recover a solution in x-space, you need a map back. Without invertibility, the lifted solution has no unique pre-image.

This is not merely a practical inconvenience — it is a topological obstruction. A homeomorphism (continuous bijection) cannot map a space with multiple disconnected local minima to a convex set, because topology is preserved under bijection. If the original problem has k local minima, any invertible lift must also have k distinct regions, precluding global convexity. For the protein folding landscape, where the number of local minima is astronomical, this obstruction is particularly severe.

---

## 4. Relaxing Invertibility: The Decoder Hypothesis

The key insight of this framework is that invertibility of the encoder is not necessary if a separate mechanism handles translation from the lifted space back to the original space. Dropping invertibility removes the topological obstruction entirely and substantially changes the character of the problem.

### 4.1 Why the Obstruction Weakens

A non-injective encoder can collapse multiple local minima into a single basin in the lifted space — it acts as a learned quotient map. For the protein conformational landscape, configurations that are equivalent under symmetry (molecular orientations, permutation of equivalent atoms) can be identified and collapsed, removing a major source of non-convexity. There is no corresponding operation available to an invertible map.

Without the bijection constraint, the encoder can also freely embed into arbitrarily high dimensions. In high dimensions it becomes generically easier to find convex arrangements of the image — the spirit of the classical kernel trick, now applied to the geometry of the energy landscape rather than just the classifier boundary.

### 4.2 Where the Difficulty Migrates

Removing invertibility from the encoder does not eliminate the difficulty — it relocates it. In a traditional optimization pipeline, one would need a decoder $\psi$ such that the composed objective $f(\psi(z))$ is convex in $z$. This is a joint condition on the decoder architecture and the original objective $f$.

We propose an alternative: rather than requiring an explicit decoder, use a *learned optimizer* (an RL agent) that receives gradient information from the lifted convex space and learns to translate those signals into effective steps on the torus. The agent implicitly learns the decoder through its policy. In the protein folding analogy, the convex surrogate provides a learned approximation of the folding funnel — a smooth, globally informative signal — while the RL agent learns a folding policy: a strategy for navigating the torsion-angle landscape that is guided by this signal but evaluated against the true energy.

---

## 5. The Proposed Framework: RL-Based Optimization with Hidden Convex Oracles

### 5.1 Problem Construction

We construct a family of synthetic optimization problems with the following structure, designed to capture the essential features of the protein conformational search problem: a low-dimensional non-convex landscape admitting a hidden convex lift.

**Visible space (non-convex).** The optimizer operates in $z \in \mathbb{R}^d$ (the "visible" or natural parameter space), where $d$ is small. In the current meta-optimizer rerun, $d = 2$, which lets us render the full visible-space energy landscape as a heatmap for qualitative inspection. This mirrors the two torsion angles $(\phi, \psi)$ of alanine dipeptide, the standard minimal benchmark for conformational search methods.

**Hidden space (convex).** A non-invertible map $F: \mathbb{R}^d \to \mathbb{R}^D$ (where $D \gg d$) lifts visible-space points into a high-dimensional "hidden" space. In the current rerun, $D = 200$. In hidden space, the objective is convex by construction:

$$E(z) = \frac{1}{2} \|F(z) - s^*\|^2$$

where $s^* = F(z^*)$ is the image of the (unknown) global minimizer $z^*$ under $F$. While $E$ is convex as a function of $s = F(z)$ (it is a simple squared distance), it is non-convex as a function of $z$ because $F$ is a nonlinear map.

**The lifting map $F$.** Each problem instance draws $F$ from a parametric family. We use Fourier-basis maps of the form:

$$F(z) = Az + a \odot \sin(Wz + b) + c \odot \cos(Vz + d)$$

where $A, W, V \in \mathbb{R}^{D \times d}$ are weight matrices with $W, V$ having integer-valued frequencies drawn from $\{1, \ldots, K\}$ with random signs, and $a, b, c, d$ are amplitude and phase parameters. The basis complexity parameter $K$ controls the frequency content and thus the degree of non-convexity in visible space. The Fourier structure is deliberate: molecular force fields express torsional energy as Fourier series in the dihedral angles, so this family captures the essential mathematical character of conformational energy landscapes.

### 5.2 Oracle Communication Model

The key experimental variable is the *information* available to the optimizer at each step. We define three oracle regimes:

1. **No oracle.** The agent observes only its current position $z_t$ and the (normalized) objective value $E(z_t)$.

2. **Visible-space gradient oracle.** The agent receives $\nabla_z E(z_t) = J_F(z_t)^\top (F(z_t) - s^*)$, the $d$-dimensional gradient in visible space. This is the gradient standard GD/Adam would use.

3. **Hidden-space gradient oracle.** The agent receives $\nabla_s E = F(z_t) - s^*$, the $D$-dimensional gradient in hidden space. This points directly toward the global minimum in the *convex* hidden space, regardless of the non-convex structure of $z$-space.

The visible gradient is a Jacobian-projected compression of the hidden gradient — it discards all components of the residual orthogonal to the column space of $J_F$. The agent with hidden-gradient access can implicitly learn a local pseudo-inverse of $J_F$, rather than following only its projected image — something closer in spirit to a Newton step than a gradient step.

### 5.3 RL Agent Architecture

The optimizer is a policy trained via Proximal Policy Optimization (PPO). At each step $t$ the agent receives the oracle token, normalized position $z_t / z_{\max}$, normalized objective, and elapsed horizon fraction. It outputs a $(d+1)$-dimensional action: a unit direction and a scalar step-size modulator. The policy network is a 2-layer MLP with tanh activations (hidden dimension 256), with separate policy and value heads. Training uses standard PPO hyperparameters ($\gamma = 0.99$, $\lambda_{\text{GAE}} = 0.95$, clip ratio $= 0.2$, entropy coefficient $= 0.01$).

### 5.4 Baseline Optimizers

- **Gradient Descent (GD):** cosine-annealed learning rate tuned via grid search over $\{0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2\}$.
- **Adam:** same cosine schedule, $\beta_1 = 0.9$, $\beta_2 = 0.999$, learning rate similarly tuned.
- **Random search:** sample uniformly from $[-z_{\max}, z_{\max}]^d$ at each step, keep best.

All baselines use the visible-space gradient $\nabla_z E$ and have no access to hidden-space information.

---

## 6. Experimental Setup

| Parameter | Value |
|---|---|
| Visible dimension $d$ | 2 |
| Hidden dimension $D$ | 200 |
| Coordinate limit $z_{\max}$ | 3 |
| Basis complexity $K$ | 3 |
| Map family | Fourier |
| Optimization horizon $H$ | 60 steps |
| Success threshold | 0.01 (normalized objective) |
| Number of seeds | 3 |

Each seed trains the RL agents on randomly drawn problem instances and evaluates on 500 fresh tasks, yielding 1,500 evaluation tasks per method across seeds.

---

## 7. Results

### 7.1 Meta-Optimizer Study: Single-Trajectory Quality

| Method | Final Objective (mean ± 95% CI) | Best Objective (mean ± 95% CI) |
|---|---|---|
| GD (tuned) | 0.4849 ± 0.0061 | 0.4849 ± 0.0061 |
| Adam (tuned) | 0.4745 ± 0.0051 | 0.4743 ± 0.0052 |
| Random search | 0.1892 ± 0.0061 | 0.1892 ± 0.0061 |
| RL (no oracle) | 0.5756 ± 0.0221 | 0.4133 ± 0.0167 |
| RL (visible gradient) | 0.4952 ± 0.0079 | 0.4392 ± 0.0096 |
| **RL (hidden gradient)** | **0.0838 ± 0.0345** | **0.0495 ± 0.0310** |

Key findings:

1. **Hidden-gradient PPO is decisively best.** It reaches 0.0838 final objective — 5.7× lower than Adam and 2.3× lower than random search.
2. **The advantage is specific to the hidden oracle.** Visible-gradient PPO (0.4952) does not beat tuned Adam, ruling out "RL is just a better optimizer" as an explanation. Only access to the convex hidden-space gradient produces a decisive gain.
3. **GD and Adam reliably get stuck.** Both finish near 0.48–0.49, *worse* than random search (0.1892), validating that the problem family is genuinely hard for gradient methods in visible space.
4. **Separation is early and persistent.** By step 10, hidden-gradient PPO is at 0.2949 versus 0.4131 for random search and 0.5063 for Adam.

### 7.2 Trajectory Geometry

The heatmap trajectories make the decoder interpretation concrete. The hidden-gradient policy crosses visible-space contours toward the reference minimum, while GD/Adam and weaker PPO variants stall, meander, or commit to the wrong basin. This is consistent with the policy having learned a local pseudo-inverse of $J_F$ — using the full D-dimensional residual to infer a step direction invisible to the projected visible-space gradient. In the protein folding analogy, the agent has learned to follow the funnel rather than getting trapped in metastable conformations.

### 7.3 Effect of Hidden Dimension: A Theoretical Analysis

The hidden-gradient agent fails to achieve useful performance at $D = 20$, despite $D = 20 > d = 2$. This is counterintuitive — a smaller hidden space might seem less complex, not more. The failure is explained by the *informativeness* of the hidden gradient, not the complexity of the map.

**The information matrix.** The local information the hidden gradient provides is captured by the Fisher information matrix:

$$G(z) = J_F(z)^\top J_F(z) \in \mathbb{R}^{d \times d}$$

Each row of $J_F$ contributes one rank-1 outer product to $G$. For the oracle to reliably identify a direction toward $z^*$, $G(z)$ must be well-conditioned — both eigenvalues large and roughly equal. Ill-conditioning means some directions in visible space are effectively invisible to the oracle.

**Distinct frequency vectors.** With $K = 3$ and $d = 2$, each row of $W$ is a vector $(w_1, w_2)$ with $w_j \in \{\pm 1, \pm 2, \pm 3\}$. The number of distinct frequency vectors is:

$$(2K)^d = 6^2 = 36$$

This is the key quantity. The contribution of each channel $i$ to $G$ is proportional to $w_i w_i^\top$. If all 36 distinct frequency vectors are represented, their contributions sum to a matrix proportional to $I_2$ by symmetry — the set $\{\pm 1, \pm 2, \pm 3\}^2$ is symmetric under sign flips, so off-diagonal terms cancel and diagonal terms are equal. Complete frequency coverage is therefore the condition for an isotropic, well-conditioned information matrix.

**Coverage as a function of D.** With D channels drawn i.i.d. uniformly from 36 possible frequency vectors, the expected number of distinct vectors represented is:

$$D_{\text{unique}}(D) = 36 \left(1 - \left(\frac{35}{36}\right)^D\right)$$

| D | Expected distinct frequencies | Coverage |
|---|---|---|
| 20 | 15.5 | 43% |
| 36 | 22.9 | 64% |
| 72 | 31.3 | 87% |
| 130 | 35.1 | 97% |
| 200 | 35.9 | ~100% |

At $D = 20$, fewer than half the frequency directions are represented in expectation. The information matrix is far from isotropic, meaning some visible-space directions receive little gradient signal from the oracle. At $D = 200$, coverage is essentially complete and the information matrix approaches the isotropic ideal.

**The predicted threshold.** The expected draws required to collect all 36 distinct frequency vectors (the coupon collector problem) is:

$$D^* = 36 \cdot H_{36} = 36 \sum_{k=1}^{36} \frac{1}{k} \approx 36 \times 4.1746 \approx 150.3$$

So the exact coupon-collector expectation is about **150** channels. The general asymptotic formula for arbitrary $K$ and $d$ is:

$$D^* \approx (2K)^d \left(\ln\!\left((2K)^d\right) + \gamma\right)$$

where $\gamma \approx 0.577$ is the Euler-Mascheroni constant. For $K = 3$, $d = 2$, this gives

$$D^* \approx 36(\ln 36 + \gamma) \approx 149.8$$

A cruder approximation drops the $\gamma$ term and gives $36 \ln 36 \approx 129$, which underestimates the true threshold. This predicts three regimes:

- **$D < 36$:** Sparse coverage, unreliable — consistent with $D = 20$ failing.
- **$D \in [36, 150]$:** Partial coverage, marginal and improving performance.
- **$D > 150$:** Near-complete coverage, reliable — consistent with $D = 200$ working.

**Testable predictions.** A sweep over $D \in \{20, 36, 50, 75, 100, 150, 200\}$ should reveal a phase transition in that range. A further controlled experiment — setting $K = 1$ at $D = 200$ — would collapse all 200 channels to the same low-frequency direction, giving effectively one distinct frequency vector and predicted performance degradation to near the $D = 20$ level. This would confirm the failure mode is frequency coverage rather than policy architecture.

---

## 8. Connection to the Convex Lift Framework

**The encoder is $F$ (non-invertible).** The Fourier map $F: \mathbb{R}^2 \to \mathbb{R}^{200}$ is a non-injective encoder lifting the visible space into a hidden space where the objective is convex. It is non-invertible by design ($D \gg d$), so the topological obstruction from Section 3 does not apply.

**The RL policy is the learned decoder.** Rather than training an explicit decoder network $\psi: \mathbb{R}^D \to \mathbb{R}^d$, the RL agent learns a policy translating hidden-space gradient information into effective steps in the original non-convex space. Given $\nabla_s E$ (pointing toward the minimum in convex hidden space), the policy outputs $\Delta z$ (a step toward the minimum in non-convex visible space). This is a *sequential* decoder — unfolding across the optimization trajectory rather than producing a single-shot inverse.

**The protein folding interpretation.** In the conformational search setting, the RL agent is learning a folding policy on the torsion-angle torus. The convex surrogate provides a smooth, globally informative signal analogous to the folding funnel — the empirically observed property that natural protein energy landscapes are biased toward the native state. The agent does not need to invert the lift; it needs to learn how to translate the funnel's gradient signal into productive torsion-angle updates from any starting conformation. This is precisely what each protein molecule does every time it folds.

**Sufficient hidden dimension is necessary.** The analysis in Section 7.3 adds a new condition to the framework: the hidden space must be large enough relative to the frequency content of $F$ for the oracle signal to be informative. This is the analog of the measurement sufficiency condition in compressed sensing and phase retrieval.

---

## 9. Relationship to Existing Work

| Method | Encoder Invertible? | Convexity Guaranteed? | Decoder | Optimizer |
|---|---|---|---|---|
| Analytic SDP lifts (torus relaxation) | No | Yes, with conditions | Rank-1 projection | Convex solver |
| ICNN (Amos 2017) | N/A | Yes (network is convex) | No | GD on ICNN output |
| Normalizing flows + ICNN | Yes (required) | Approximately | Implicit (inverse flow) | GD |
| Koopman networks | Approximately | For dynamics, not energy | Yes | Linear predictor |
| **This work** | **No (relaxed)** | **In hidden space (by construction)** | **Learned RL policy** | **PPO agent** |

---

## 10. Limitations and Future Directions

**Known map family.** The lifting map $F$ is sampled from a known parametric Fourier family. In the protein setting, the lift would need to be learned jointly with the optimizer from force field evaluations or molecular dynamics data.

**Known oracle.** The hidden-gradient oracle provides the exact convex gradient. In practice, this signal would need to be approximated or learned — for instance, from an ICNN trained on sampled conformational energies.

**Low visible dimension.** With $d = 2$, random search is competitive. The advantage of learned optimization is expected to grow substantially in higher dimensions. The immediate next benchmark is alanine dipeptide itself ($d = 2$, periodic, with a known free energy surface for ground-truth comparison). Beyond that, scaling to short peptides ($d = 10$–$20$ torsion angles) and eventually full proteins ($d = 100$+) is the central challenge.

**Hidden dimension threshold.** The analysis in Section 7.3 predicts a phase transition near $D^* \approx 145$ for the current $K = 3$, $d = 2$ setup. Performance at intermediate D values and the effect of varying K at fixed D remain to be characterized experimentally.

**Limited seed count.** Three seeds leaves the key result with wide confidence intervals ($0.0838 \pm 0.0345$).

**Toward a complete pipeline.** Immediate next steps are: (1) training $F$ end-to-end with a Hessian eigenvalue penalty to encourage hidden-space convexity; (2) replacing the exact gradient oracle with a learned bandwidth-limited communication channel; (3) validating on alanine dipeptide as the first physically realistic test case; (4) scaling to $d = 10$–$100$ visible dimensions corresponding to multi-residue peptide torsion-angle spaces.

---

## 11. Summary

We present empirical evidence that reinforcement learning agents can learn to optimize non-convex objectives by exploiting hidden convex structure communicated through an oracle. Problems are constructed that are non-convex in a 2D visible space but convex in a 200D hidden space, connected by a non-invertible Fourier lifting map — a synthetic analog of the protein conformational search problem, where torsional energy landscapes on the torus are non-convex but admit convex reformulations in higher-dimensional PSD spaces.

The hidden-gradient PPO agent reaches 0.0838 ± 0.0345 final objective across 1,500 evaluation tasks, versus 0.4745 for Adam and 0.1892 for random search. The advantage is specific to the hidden oracle — visible-gradient PPO does not beat tuned Adam — confirming that access to the convex structure of the lifted space is the operative mechanism.

A theoretical analysis of the hidden dimension requirement shows that the relevant quantity is not D itself but the coverage of the $(2K)^d = 36$ distinct frequency directions in the Fourier map. The coupon collector threshold:

$$D^* \approx (2K)^d \ln(2K)^d \approx 145$$

predicts a phase transition in performance between $D = 20$ (failing) and $D = 200$ (working), with a testable intermediate regime at $D \in [36, 145]$. This provides a quantitative and experimentally falsifiable condition on when hidden convex structure is exploitable by a learned optimizer.

The framework reframes the protein folding problem as an RL task: rather than searching the torsion-angle landscape directly, learn a policy that translates gradient information from a convex lifted space into effective conformational updates. The RL agent learns a folding strategy — a sequential decoder that navigates the non-convex torus guided by a convex oracle — mirroring the funnel-guided process by which real proteins fold reliably from arbitrary initial configurations.
