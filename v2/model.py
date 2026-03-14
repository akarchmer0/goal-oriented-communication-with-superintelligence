import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        token_feature_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        architecture: str = "mlp",
        action_space_type: str = "discrete",
        continuous_init_std: float = 0.4,
    ):
        super().__init__()
        if architecture not in {"mlp", "gru"}:
            raise ValueError(f"Unsupported architecture '{architecture}'")
        if action_space_type not in {"discrete", "continuous"}:
            raise ValueError(f"Unsupported action_space_type '{action_space_type}'")
        if continuous_init_std <= 0.0:
            raise ValueError("continuous_init_std must be > 0")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.action_space_type = action_space_type

        feature_dim = token_feature_dim + 2
        if architecture == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
        else:
            self.backbone = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        if self.action_space_type == "continuous":
            init = torch.full((action_dim,), float(continuous_init_std), dtype=torch.float32)
            self.log_std = nn.Parameter(torch.log(init))
        else:
            self.log_std = None
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.architecture != "gru":
            return None
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)

    def _encode(
        self,
        token_features: torch.Tensor,
        dist_feature: torch.Tensor,
        step_fraction: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = torch.cat(
            [token_features, dist_feature.unsqueeze(-1), step_fraction.unsqueeze(-1)],
            dim=-1,
        )
        if self.architecture == "mlp":
            return self.backbone(features), None

        batch_size = features.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size,
                self.hidden_dim,
                dtype=features.dtype,
                device=features.device,
            )
        gru_out, gru_state = self.backbone(features.unsqueeze(1), hidden_state.unsqueeze(0))
        return gru_out.squeeze(1), gru_state.squeeze(0)

    def forward(
        self,
        token_features: torch.Tensor,
        dist_feature: torch.Tensor,
        step_fraction: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden, next_hidden_state = self._encode(
            token_features,
            dist_feature,
            step_fraction,
            hidden_state,
        )
        policy_output = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return policy_output, value, next_hidden_state

    def _distribution(self, policy_output: torch.Tensor):
        if self.action_space_type == "discrete":
            return Categorical(logits=policy_output)
        assert self.log_std is not None
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(policy_output)
        return Normal(loc=policy_output, scale=std)

    @torch.no_grad()
    def act(
        self,
        token_features: torch.Tensor,
        dist_feature: torch.Tensor,
        step_fraction: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        policy_output, value, next_hidden_state = self.forward(
            token_features,
            dist_feature,
            step_fraction,
            hidden_state=hidden_state,
        )
        distribution = self._distribution(policy_output)
        if self.action_space_type == "discrete":
            if deterministic:
                action = torch.argmax(policy_output, dim=-1)
            else:
                action = distribution.sample()
            logprob = distribution.log_prob(action)
        else:
            if deterministic:
                action = policy_output
            else:
                action = distribution.rsample()
            # Keep continuous-policy objective approximately invariant to action dimension.
            logprob = distribution.log_prob(action).mean(dim=-1)
        return action, logprob, value, next_hidden_state

    def evaluate_actions(
        self,
        token_features: torch.Tensor,
        dist_feature: torch.Tensor,
        step_fraction: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_output, value, _ = self.forward(
            token_features,
            dist_feature,
            step_fraction,
            hidden_state=hidden_state,
        )
        distribution = self._distribution(policy_output)
        if self.action_space_type == "discrete":
            logprob = distribution.log_prob(actions)
            entropy = distribution.entropy()
        else:
            # Match act(): average across dimensions to avoid scaling pressure with action_dim.
            logprob = distribution.log_prob(actions).mean(dim=-1)
            entropy = distribution.entropy().mean(dim=-1)
        return logprob, entropy, value
