# Spatial Goal-Oriented Communication: Experiment Report

## 1. Introduction

This report describes the **spatial experiment** from the Goal-Oriented Communication project. The experiment investigates whether a reinforcement learning agent can learn to interpret continuous oracle messages to solve an optimization problem that it cannot solve by local information alone.

The core question is:

> *Can a policy learn to use high-dimensional gradient signals from an oracle to navigate a low-dimensional control space, when the objective lives in a hidden space the agent cannot directly observe?*

## 2. Motivation

### 2.1 From Discrete Navigation to Continuous Optimization

The broader project studies communication between an oracle and an agent. In the original **graph task**, an agent navigates a discrete directed graph toward a target node, guided by discrete cipher tokens from the oracle. The agent must learn the cipher—a mapping from tokens to actions—purely from reward feedback.

The spatial experiment extends this idea to a **continuous** setting. Instead of choosing among discrete neighbors, the agent takes continuous 2D actions. Instead of discrete cipher tokens, the oracle sends continuous gradient vectors. The fundamental challenge remains the same: the agent must learn to interpret the oracle's messages and translate them into useful actions.

### 2.2 The Hidden-Space Narrative

The spatial experiment is motivated by a scenario where:

- A **hidden objective** \( E(s) = \frac{1}{2} \| s - s^* \|^2 \) is defined over a high-dimensional space \( s \in \mathbb{R}^D \), and is **convex** in that space.
- The agent controls only a **visible 2D coordinate** \( z \in \mathbb{R}^2 \).
- The hidden state is determined by a **nonlinear map** \( s = F(z) \), so the energy landscape \( E(F(z)) \) projected onto the 2D control space is **non-convex** with many local minima.
- An oracle observes the hidden space and communicates gradient information \( g_t = s_t - s^* \) to the agent at each step.

The agent's task is to minimize \( E(F(z)) \) by choosing 2D actions, using the oracle's hidden-space gradient messages as guidance. The difficulty is that the oracle's messages live in \( \mathbb{R}^D \) and describe a space the agent never directly sees—the agent must learn to map these signals into useful 2D movements.

### 2.3 Why This Matters

This setup tests whether learned communication can outperform standard optimization. A natural baseline is **gradient descent in the visible 2D space**, using the true gradient \( \nabla_z E = J^\top (s - s^*) \), where \( J \) is the Jacobian of \( F \). This baseline has full access to the true 2D gradient, which requires computing the Jacobian—information the RL agent does not have. The question is whether the RL agent, given only the raw hidden-space gradient from the oracle, can learn a policy that outperforms this analytically-informed baseline.

## 3. Experimental Setup

### 3.1 Environment

**Control space.** The agent operates in \( z \in [-8, 8]^2 \).

**Hidden map \( F \).** The map \( F: \mathbb{R}^2 \to \mathbb{R}^D \) uses a Fourier basis:

\[
F(z) = A z + a \odot \sin(W z + b) + c \odot \cos(V z + d)
\]

where \( A \) is a learned linear component, \( W, V \) have integer frequencies sampled from \([1, 4]\) with random signs, and \( a, b, c, d \) are random amplitudes and phases. This creates a rich, nonlinear landscape with many local minima in the 2D projection. The map parameters are sampled once and shared across all parallel environments within a run.

**Hidden dimension.** \( D = 100 \). The hidden space is 50x the dimensionality of the visible space, making the task substantially harder than if the agent could directly optimize in \( \mathbb{R}^D \).

**Objective.** \( E(s) = \frac{1}{2} \| s - s^* \|^2 \), where \( s^* = F(z^*) \) for a randomly sampled reference minimum \( z^* \). This is convex in the hidden space but non-convex when projected to the 2D control surface.

**Actions.** The policy outputs a 3D continuous vector \( (d_x, d_y, \text{step\_raw}) \). The direction \( (d_x, d_y) \) is normalized to a unit vector, and \( \text{step\_raw} \) is passed through a sigmoid to produce a step scale in \( [0, 1] \). The agent moves:

\[
z_{t+1} = \text{clip}\!\big(z_t + 0.25 \cdot \sigma(\text{step\_raw}) \cdot \hat{d},\; [-8, 8]\big)
\]

**Episodes.** Each episode starts from a random \( z_0 \in [-8, 8]^2 \) with a random target minimum \( z^* \). The horizon is 128 steps. An episode succeeds if \( E(F(z)) \leq 0.02 \).

**Reward (S0 sensing).** At each step:
\[
r_t = \text{clip}(E(F(z_t)) - E(F(z_{t+1})),\; -1,\; 1) + \mathbb{1}[\text{success}]
\]

The agent is rewarded for reducing the objective, with a +1 bonus upon success.

### 3.2 Oracle

The oracle operates in **convex_gradient** mode: at each step, it sends the full hidden-space gradient \( g_t = s_t - s^* \in \mathbb{R}^{100} \) to the agent. This vector points from the current hidden state toward the target in the hidden space, but the agent must learn how this 100-dimensional signal relates to useful 2D movements.

### 3.3 Observation Space

The policy receives:
- **Token features** (dim 102): the oracle gradient \( g_t \in \mathbb{R}^{100} \) concatenated with the normalized position \( z_t / 8 \in \mathbb{R}^2 \).
- **Distance feature** (dim 1): the normalized objective \( E(F(z_t)) / E_{\max} \).
- **Step fraction** (dim 1): the fraction of the horizon elapsed \( t / H \).

### 3.4 Model Architecture

The policy and value function share a 2-layer MLP backbone (256 hidden units, Tanh activations). The total input dimension is 104 (102 token features + 1 distance + 1 step fraction).

- **Policy head:** Linear layer to 3 outputs (direction + step scale), parameterizing a diagonal Gaussian with learnable log-standard-deviation (initialized at 0.4).
- **Value head:** Linear layer to a scalar state-value estimate.

### 3.5 Training

Training uses **Proximal Policy Optimization (PPO)** with:

| Parameter | Value |
|-----------|-------|
| Parallel environments | 32 |
| Rollout length | 64 steps |
| PPO epochs per update | 4 |
| Minibatches per epoch | 4 |
| Discount \( \gamma \) | 0.99 |
| GAE \( \lambda \) | 0.95 |
| Clip ratio \( \epsilon \) | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Learning rate | 3 × 10⁻⁴ (Adam) |
| Max gradient norm | 0.5 |
| Total environment steps | 300,000 |

### 3.6 Baselines

Two baselines are evaluated on each episode, starting from the same initial position \( z_0 \):

1. **2D Gradient Descent (GD).** Uses the true visible-space gradient \( \nabla_z E = J^\top (s - s^*) \), computed analytically via the Jacobian of \( F \). The step size is cosine-annealed over the horizon. This baseline has strictly more information than the RL agent—it has access to the Jacobian, which encodes the local geometry of the map \( F \).

2. **PPO without oracle.** A second PPO policy trained identically but with zero oracle messages. This ablation isolates the value of the oracle's gradient information.

## 4. Results

Results are reported from two independent runs (seeds 1705 and 2626), each trained for 300,000 environment steps (~2,340 episodes).

### 4.1 Final Performance

| Metric | Learned (oracle) | 2D Gradient Descent | PPO (no oracle) |
|--------|:-----------------:|:-------------------:|:---------------:|
| Final objective \( E(F(z)) \) | **93.9 ± 1.6** | 181.9 ± 2.5 | 178.1 ± 18.2 |
| Distance to reference min | **1.90 ± 0.09** | 7.90 ± 0.17 | 6.42 ± 0.21 |

*(Values are mean ± half-range across two seeds, using a 100-episode running average at the end of training.)*

The learned policy with oracle guidance achieves:
- **~2x lower objective** than both the 2D gradient descent baseline and the no-oracle PPO baseline.
- **~4x closer** to the reference minimum in Euclidean distance compared to gradient descent, and **~3.4x closer** than no-oracle PPO.

### 4.2 Learning Dynamics

The learning curves show three distinct phases:

1. **Early exploration (episodes 0–200).** The policy performs poorly, with objective values around 280–320—worse than both baselines. The policy has not yet learned to interpret the oracle's gradient messages.

2. **Rapid improvement (episodes 200–800).** The policy discovers how to use the oracle signals and rapidly improves, crossing below both baselines by approximately episode 300–400.

3. **Convergence (episodes 800+).** The policy stabilizes at objective values around 90–100, with continued slow improvement. Both baselines remain flat at their respective levels (~180 for GD, ~160–200 for no-oracle PPO).

### 4.3 Why the Learned Policy Outperforms Gradient Descent

The 2D gradient descent baseline follows \( -\nabla_z E \), the steepest descent direction in the visible space. While this direction is locally optimal, the non-convex landscape of \( E(F(z)) \) means that steepest descent frequently gets trapped in local minima or oscillates on ridges.

The learned policy has access to the raw hidden-space gradient \( g_t = s_t - s^* \), which is a 100-dimensional vector pointing toward the global optimum in the hidden space. By learning a mapping from \( (g_t, z_t) \) to actions, the policy effectively learns an implicit model of the relationship between the hidden-space gradient and useful 2D movements. This allows it to:

- **Avoid local minima** that trap the 2D gradient descent baseline.
- **Take globally-informed steps**, since the hidden-space gradient always points toward \( s^* \) regardless of local landscape curvature.
- **Adaptively scale step sizes** via the learned step-scale component, rather than relying on a fixed cosine annealing schedule.

### 4.4 Trajectory Visualization

The trajectory plots on the 2D energy landscape provide qualitative insight:

- **Learned policy (black):** Navigates purposefully toward low-energy regions near the reference minimum, often taking smooth, curved paths that avoid high-energy ridges.
- **2D GD baseline (red):** Takes short, locally-greedy steps that quickly stall near the starting position, trapped by the non-convex landscape.
- **No-oracle PPO (blue):** Wanders broadly across the landscape without directional guidance, occasionally finding lower-energy regions by chance but unable to consistently approach the target.

**Early training (episode 100):** The learned policy moves erratically, without clear direction—it has not yet learned to decode the oracle's messages.

**Mid training (episode 1000):** The policy has learned to move toward lower-energy regions, navigating purposefully toward the reference minimum.

**Final (episode 2300+):** The policy reliably reaches a neighborhood of the reference minimum, substantially outperforming both baselines.

### 4.5 Success Rate

Despite the strong improvement in objective value and distance, the strict success threshold (\( E \leq 0.02 \)) is rarely achieved. Most episodes run to the full 128-step horizon (average path length ~127.8). This reflects the fundamental difficulty of exactly reaching the global minimum of a non-convex landscape through a nonlinear 100-dimensional map, rather than a failure of the policy—the policy does get much closer than the baselines, but the final precision required is extremely high.

## 5. Discussion

### 5.1 Communication as Implicit Model Learning

The central finding is that the learned policy, using only the raw hidden-space gradient as a communication signal, outperforms a baseline that has access to the analytically-computed true 2D gradient. This suggests that the policy learns more than just a local descent rule—it effectively builds an implicit mapping from the hidden-space geometry to useful visible-space actions.

This is notable because the 2D gradient descent baseline has strictly more information in a traditional optimization sense: it knows the exact local slope of the objective surface. Yet the RL agent, by processing the global hidden-space gradient through a learned neural network, can extract globally useful information that local gradients miss.

### 5.2 Value of Oracle Information

Comparing the oracle-guided policy to the no-oracle PPO baseline quantifies the value of the oracle's messages. Without oracle guidance, the policy achieves objective values of ~160–196 (comparable to or slightly better than 2D GD), versus ~92–96 with oracle guidance. The oracle's hidden-space gradient provides approximately a **2x reduction** in final objective value, confirming that the agent successfully learns to extract useful information from the communication channel.

### 5.3 Limitations and Future Directions

- **Fixed map:** The Fourier map \( F \) is sampled once per run. Testing with `refresh_map_each_episode=True` would assess whether the policy can generalize across different landscape geometries.
- **Oracle fidelity:** The current oracle sends the exact hidden-space gradient. Introducing noise, dimensionality reduction (via `linear_embedding` mode), or adversarial corruption would test robustness.
- **Scalability:** The current runs use \( D = 100 \). Scaling to larger hidden dimensions or more complex map families would probe the limits of this approach.
- **Success rate:** The near-zero success rate suggests that the reward signal (energy decrease) is effective for learning approach behavior but may need augmentation (e.g., curriculum learning, finer step sizes near the minimum) to achieve precise convergence.

## 6. Conclusion

The spatial experiment demonstrates that a reinforcement learning agent can learn to interpret high-dimensional oracle messages and use them to solve a continuous optimization problem more effectively than analytically-informed gradient descent. The key insight is that communication from a hidden-space oracle, combined with learned interpretation via a neural policy, enables globally-informed navigation of a non-convex landscape—something that local gradient information alone cannot achieve.

## Appendix: Configuration

```
task:                     spatial
oracle_mode:              convex_gradient
spatial_hidden_dim:       100
spatial_visible_dim:      2
spatial_coord_limit:      8
spatial_step_size:        0.25
spatial_success_threshold: 0.02
spatial_basis_complexity:  3
spatial_f_type:           FOURIER
max_horizon:              128
hidden_dim (policy):      256
n_env:                    32
train_steps:              300,000
seeds:                    1705, 2626
```
