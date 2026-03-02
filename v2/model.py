import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        token_vocab_size: int,
        null_token_id: int,
        action_dim: int,
        token_embed_dim: int = 16,
        hidden_dim: int = 64,
        architecture: str = "mlp",
    ):
        super().__init__()
        if architecture not in {"mlp", "gru"}:
            raise ValueError(f"Unsupported architecture '{architecture}'")
        self.null_token_id = null_token_id
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture

        self.token_embedding = nn.Embedding(token_vocab_size + 1, token_embed_dim)
        if architecture == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(token_embed_dim + 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
        else:
            self.backbone = nn.GRU(
                input_size=token_embed_dim + 2,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.architecture != "gru":
            return None
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)

    def _encode(
        self,
        token: torch.Tensor,
        dist_feature: torch.Tensor,
        step_fraction: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedded = self.token_embedding(token)
        features = torch.cat(
            [embedded, dist_feature.unsqueeze(-1), step_fraction.unsqueeze(-1)],
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
        token: torch.Tensor,
        dist_feature: torch.Tensor,
        step_fraction: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden, next_hidden_state = self._encode(token, dist_feature, step_fraction, hidden_state)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value, next_hidden_state

    @torch.no_grad()
    def act(
        self,
        token: torch.Tensor,
        dist_feature: torch.Tensor,
        step_fraction: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        logits, value, next_hidden_state = self.forward(
            token,
            dist_feature,
            step_fraction,
            hidden_state=hidden_state,
        )
        distribution = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = distribution.sample()
        logprob = distribution.log_prob(action)
        return action, logprob, value, next_hidden_state

    def evaluate_actions(
        self,
        token: torch.Tensor,
        dist_feature: torch.Tensor,
        step_fraction: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, _ = self.forward(
            token,
            dist_feature,
            step_fraction,
            hidden_state=hidden_state,
        )
        distribution = Categorical(logits=logits)
        logprob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return logprob, entropy, value
