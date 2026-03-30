from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import PolicyValueNet


@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    auxiliary_coef: float = 0.0
    auxiliary_step_coef: float = 0.25
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatches: int = 4


class RolloutBuffer:
    def __init__(
        self,
        rollout_len: int,
        n_env: int,
        token_feature_dim: int,
        action_dim: int,
        action_dtype: str,
        teacher_action_dtype: str | None = None,
        teacher_action_dim: int | None = None,
    ):
        self.rollout_len = rollout_len
        self.n_env = n_env
        self.token_feature_dim = token_feature_dim
        self.action_dim = int(action_dim)
        self.action_dtype = action_dtype
        self.teacher_action_dtype = teacher_action_dtype
        self.teacher_action_dim = (
            None if teacher_action_dim is None else int(teacher_action_dim)
        )
        shape = (rollout_len, n_env)

        self.token_features = np.zeros((rollout_len, n_env, token_feature_dim), dtype=np.float32)
        self.dist_features = np.zeros(shape, dtype=np.float32)
        self.step_fractions = np.zeros(shape, dtype=np.float32)
        if self.action_dtype == "discrete":
            self.actions = np.zeros(shape, dtype=np.int64)
        elif self.action_dtype == "continuous":
            self.actions = np.zeros((rollout_len, n_env, self.action_dim), dtype=np.float32)
        else:
            raise ValueError("action_dtype must be 'discrete' or 'continuous'")
        self.logprobs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        if self.teacher_action_dtype is None:
            self.teacher_actions = None
            self.teacher_action_mask = None
        elif self.teacher_action_dtype == "discrete":
            self.teacher_actions = np.zeros(shape, dtype=np.int64)
            self.teacher_action_mask = np.zeros(shape, dtype=np.float32)
        elif self.teacher_action_dtype == "continuous":
            target_dim = int(
                self.teacher_action_dim
                if self.teacher_action_dim is not None
                else self.action_dim
            )
            self.teacher_actions = np.zeros(
                (rollout_len, n_env, target_dim), dtype=np.float32
            )
            self.teacher_action_mask = np.zeros(shape, dtype=np.float32)
        else:
            raise ValueError(
                "teacher_action_dtype must be None, 'discrete', or 'continuous'"
            )
        self.hidden_states: np.ndarray | None = None

        self.ptr = 0

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        logprob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        hidden_state: np.ndarray | None = None,
    ) -> None:
        index = self.ptr
        self.token_features[index] = obs["token_features"]
        self.dist_features[index] = obs["dist"]
        self.step_fractions[index] = obs["step_frac"]
        self.actions[index] = action
        self.logprobs[index] = logprob
        self.rewards[index] = reward
        self.dones[index] = done.astype(np.float32)
        self.values[index] = value
        if hidden_state is not None:
            if self.hidden_states is None:
                self.hidden_states = np.zeros(
                    (self.rollout_len, self.n_env, hidden_state.shape[-1]),
                    dtype=np.float32,
                )
            self.hidden_states[index] = hidden_state
        if self.teacher_actions is not None and "teacher_action" in obs:
            self.teacher_actions[index] = obs["teacher_action"]
            if self.teacher_action_mask is not None:
                self.teacher_action_mask[index] = obs.get(
                    "teacher_action_mask",
                    np.ones(self.n_env, dtype=np.float32),
                )
        self.ptr += 1

    def reset(self) -> None:
        self.ptr = 0
        self.hidden_states = None


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rollout_len, n_env = rewards.shape
    # Use float64 to prevent overflow in the recursive accumulation
    advantages = np.zeros((rollout_len, n_env), dtype=np.float64)
    last_gae = np.zeros(n_env, dtype=np.float64)
    rewards64 = rewards.astype(np.float64)
    values64 = values.astype(np.float64)
    dones64 = dones.astype(np.float64)
    last_values64 = last_values.astype(np.float64)

    for step in reversed(range(rollout_len)):
        if step == rollout_len - 1:
            next_values = last_values64
        else:
            next_values = values64[step + 1]

        non_terminal = 1.0 - dones64[step]
        delta = rewards64[step] + gamma * next_values * non_terminal - values64[step]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[step] = last_gae

    returns = advantages + values64
    return advantages.astype(np.float32), returns.astype(np.float32)


def ppo_update(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    hparams: PPOHyperParams,
    device: torch.device,
) -> dict[str, float]:
    batch_size = buffer.rollout_len * buffer.n_env
    minibatch_size = max(1, batch_size // hparams.minibatches)

    token_features = torch.as_tensor(
        buffer.token_features.reshape(-1, buffer.token_feature_dim),
        device=device,
        dtype=torch.float32,
    )
    dist_feature = torch.as_tensor(
        buffer.dist_features.reshape(-1), device=device, dtype=torch.float32
    )
    step_fraction = torch.as_tensor(
        buffer.step_fractions.reshape(-1), device=device, dtype=torch.float32
    )
    if buffer.action_dtype == "discrete":
        actions = torch.as_tensor(buffer.actions.reshape(-1), device=device, dtype=torch.long)
    else:
        actions = torch.as_tensor(
            buffer.actions.reshape(-1, buffer.action_dim),
            device=device,
            dtype=torch.float32,
        )
    old_logprobs = torch.as_tensor(
        buffer.logprobs.reshape(-1), device=device, dtype=torch.float32
    )
    advantages_t = torch.as_tensor(advantages.reshape(-1), device=device, dtype=torch.float32)
    returns_t = torch.as_tensor(returns.reshape(-1), device=device, dtype=torch.float32)
    hidden_state = None
    if buffer.hidden_states is not None:
        hidden_state = torch.as_tensor(
            buffer.hidden_states.reshape(-1, buffer.hidden_states.shape[-1]),
            device=device,
            dtype=torch.float32,
        )
    teacher_actions = None
    teacher_action_mask = None
    if buffer.teacher_actions is not None:
        if buffer.teacher_action_dtype == "discrete":
            teacher_actions = torch.as_tensor(
                buffer.teacher_actions.reshape(-1), device=device, dtype=torch.long
            )
        else:
            teacher_actions = torch.as_tensor(
                buffer.teacher_actions.reshape(-1, buffer.teacher_actions.shape[-1]),
                device=device,
                dtype=torch.float32,
            )
        if buffer.teacher_action_mask is not None:
            teacher_action_mask = torch.as_tensor(
                buffer.teacher_action_mask.reshape(-1),
                device=device,
                dtype=torch.float32,
            )

    # Guard against NaN — skip update entirely if data is corrupted
    if torch.isnan(advantages_t).any() or torch.isnan(returns_t).any():
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "auxiliary_loss": 0.0,
        }

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropy_values: list[float] = []
    auxiliary_losses: list[float] = []

    for _ in range(hparams.ppo_epochs):
        permutation = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            index = permutation[start : start + minibatch_size]

            new_logprob, entropy, values = model.evaluate_actions(
                token_features[index],
                dist_feature[index],
                step_fraction[index],
                actions[index],
                hidden_state[index] if hidden_state is not None else None,
            )

            ratio = torch.exp(new_logprob - old_logprobs[index])
            unclipped = ratio * advantages_t[index]
            clipped = torch.clamp(ratio, 1.0 - hparams.clip_ratio, 1.0 + hparams.clip_ratio)
            clipped_objective = clipped * advantages_t[index]
            policy_loss = -torch.mean(torch.minimum(unclipped, clipped_objective))

            value_loss = F.mse_loss(values, returns_t[index])
            entropy_bonus = torch.mean(entropy)
            auxiliary_loss = torch.zeros((), device=device)

            if (
                teacher_actions is not None
                and teacher_action_mask is not None
                and hparams.auxiliary_coef > 0.0
            ):
                mask = teacher_action_mask[index]
                raw_mask_sum = mask.sum()
                if float(raw_mask_sum.item()) > 0.0:
                    mask_sum = torch.clamp(raw_mask_sum, min=1.0)
                    policy_output, _, _ = model.forward(
                        token_features[index],
                        dist_feature[index],
                        step_fraction[index],
                        hidden_state=hidden_state[index]
                        if hidden_state is not None
                        else None,
                    )
                    if buffer.teacher_action_dtype == "discrete":
                        aux_per_item = F.cross_entropy(
                            policy_output,
                            teacher_actions[index],
                            reduction="none",
                        )
                    else:
                        target_actions = teacher_actions[index]
                        direction_dim = max(1, policy_output.shape[-1] - 1)
                        pred_dir = policy_output[..., :direction_dim]
                        target_dir = target_actions[..., :direction_dim]
                        pred_norm = pred_dir.norm(dim=-1).clamp(min=1e-8)
                        target_norm = target_dir.norm(dim=-1).clamp(min=1e-8)
                        cosine = (pred_dir * target_dir).sum(dim=-1) / (
                            pred_norm * target_norm
                        )
                        aux_per_item = 1.0 - cosine
                        if target_actions.shape[-1] > direction_dim:
                            step_loss = F.smooth_l1_loss(
                                policy_output[..., direction_dim],
                                target_actions[..., direction_dim],
                                reduction="none",
                            )
                            aux_per_item = (
                                aux_per_item
                                + hparams.auxiliary_step_coef * step_loss
                            )
                    auxiliary_loss = (aux_per_item * mask).sum() / mask_sum

            loss = (
                policy_loss
                + hparams.value_coef * value_loss
                - hparams.entropy_coef * entropy_bonus
                + hparams.auxiliary_coef * auxiliary_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Check for NaN gradients before stepping
            has_nan = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break
            if has_nan:
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_values.append(float(entropy_bonus.item()))
            auxiliary_losses.append(float(auxiliary_loss.item()))

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "auxiliary_loss": float(np.mean(auxiliary_losses))
        if auxiliary_losses
        else 0.0,
    }
