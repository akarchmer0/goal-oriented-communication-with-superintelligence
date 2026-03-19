from dataclasses import dataclass

import numpy as np

from .distances import INF_DISTANCE
from .oracle import Oracle


@dataclass
class EpisodeSpec:
    source: int
    target_index: int
    target_node: int
    shortest_dist: int
    horizon: int
    message_tokens: np.ndarray


class VectorizedGraphEnv:
    def __init__(
        self,
        out_neighbors: np.ndarray,
        target_nodes: np.ndarray,
        dist_pool: np.ndarray,
        oracle: Oracle,
        n_env: int,
        sensing: str,
        max_horizon: int,
        seed: int,
        s1_step_penalty: float = -0.01,
        reward_noise_std: float = 0.0,
    ):
        if sensing not in {"S0", "S1"}:
            raise ValueError("sensing must be S0 or S1")
        if reward_noise_std < 0.0:
            raise ValueError("reward_noise_std must be >= 0")

        self.out_neighbors = out_neighbors
        self.target_nodes = target_nodes
        self.dist_pool = dist_pool
        self.oracle = oracle
        self.n_env = n_env
        self.sensing = sensing
        self.max_horizon = max_horizon
        self.s1_step_penalty = s1_step_penalty
        self.reward_noise_std = float(reward_noise_std)
        self.rng = np.random.default_rng(seed)

        self.n = out_neighbors.shape[0]
        self.d = out_neighbors.shape[1]
        self.t_pool_size = target_nodes.shape[0]
        self.null_token_id = oracle.token_vocab_size
        self.token_feature_dim = oracle.token_vocab_size + 1
        self.max_dist_norm = float(self.n)
        self.unreachable_penalty_dist = float(2 * self.n)

        self.reachable_sources_by_target: list[np.ndarray] = []
        valid_targets: list[int] = []
        for target_index in range(self.t_pool_size):
            dist_row = self.dist_pool[target_index]
            reachable_sources = np.where((dist_row > 0) & (dist_row < INF_DISTANCE))[0]
            self.reachable_sources_by_target.append(reachable_sources)
            if reachable_sources.size > 0:
                valid_targets.append(target_index)

        if not valid_targets:
            raise RuntimeError(
                "No valid targets in target pool: every target has zero reachable non-target sources"
            )
        self.valid_target_indices = np.asarray(valid_targets, dtype=np.int32)

        self.current_nodes = np.zeros(n_env, dtype=np.int32)
        self.target_indices = np.zeros(n_env, dtype=np.int32)
        self.steps = np.zeros(n_env, dtype=np.int32)
        self.horizons = np.ones(n_env, dtype=np.int32)
        self.initial_dist = np.ones(n_env, dtype=np.int32)
        self.messages: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_env)]
        self.completed_episodes = 0

        for index in range(self.n_env):
            self._reset_env(index)

    def _sample_target_index(self) -> int:
        chosen = self.rng.choice(self.valid_target_indices)
        return int(chosen)

    def _choose_shortest_action(self, node: int, dist_to_target: np.ndarray) -> int:
        neighbors = self.out_neighbors[node]
        current_dist = int(dist_to_target[node])
        target_dist = current_dist - 1

        candidates = np.where(dist_to_target[neighbors] == target_dist)[0]
        if candidates.size > 0:
            return int(self.rng.choice(candidates))

        finite_mask = dist_to_target[neighbors] < INF_DISTANCE
        if np.any(finite_mask):
            finite_indices = np.where(finite_mask)[0]
            best_local = finite_indices[np.argmin(dist_to_target[neighbors][finite_mask])]
            return int(best_local)

        return int(self.rng.integers(0, self.d))

    def _oracle_actions_for_episode(self, source: int, target_index: int) -> np.ndarray:
        dist_to_target = self.dist_pool[target_index]
        shortest_dist = int(dist_to_target[source])
        if shortest_dist <= 0 or shortest_dist >= INF_DISTANCE:
            return np.empty(0, dtype=np.int64)

        node = source
        actions: list[int] = []
        for _ in range(shortest_dist):
            if node == int(self.target_nodes[target_index]):
                break
            action = self._choose_shortest_action(node, dist_to_target)
            actions.append(action)
            node = int(self.out_neighbors[node, action])

        return np.asarray(actions, dtype=np.int64)

    def sample_episode_spec(self) -> EpisodeSpec:
        target_index = self._sample_target_index()
        dist_to_target = self.dist_pool[target_index]
        source = int(self.rng.choice(self.reachable_sources_by_target[target_index]))
        shortest_dist = int(dist_to_target[source])
        oracle_actions = self._oracle_actions_for_episode(source, target_index)
        message_tokens = self.oracle.encode_actions(oracle_actions, self.rng)
        horizon = max(1, min(2 * max(1, shortest_dist), self.max_horizon))

        return EpisodeSpec(
            source=source,
            target_index=target_index,
            target_node=int(self.target_nodes[target_index]),
            shortest_dist=shortest_dist,
            horizon=horizon,
            message_tokens=message_tokens,
        )

    def _reset_env(self, env_index: int) -> None:
        spec = self.sample_episode_spec()
        self.current_nodes[env_index] = spec.source
        self.target_indices[env_index] = spec.target_index
        self.steps[env_index] = 0
        self.horizons[env_index] = spec.horizon
        self.initial_dist[env_index] = spec.shortest_dist
        self.messages[env_index] = spec.message_tokens

    def _get_tokens(self) -> np.ndarray:
        tokens = np.empty(self.n_env, dtype=np.int64)
        for env_index in range(self.n_env):
            step = int(self.steps[env_index])
            message = self.messages[env_index]
            if self.oracle.mode == "no_oracle":
                tokens[env_index] = self.null_token_id
            elif step < message.shape[0]:
                tokens[env_index] = int(message[step])
            else:
                tokens[env_index] = self.null_token_id
        return tokens

    def get_obs(self) -> dict[str, np.ndarray]:
        tokens = self._get_tokens()
        token_features = np.zeros((self.n_env, self.token_feature_dim), dtype=np.float32)
        token_features[np.arange(self.n_env), tokens] = 1.0
        d_cur = self.dist_pool[self.target_indices, self.current_nodes]
        dist_feature = np.where(
            d_cur < INF_DISTANCE,
            np.minimum(d_cur.astype(np.float32), self.max_dist_norm) / self.max_dist_norm,
            1.0,
        ).astype(np.float32)
        step_fraction = (self.steps / np.maximum(self.horizons, 1)).astype(np.float32)

        return {
            "token": tokens,
            "token_features": token_features,
            "dist": dist_feature,
            "step_frac": step_fraction,
        }

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        rewards = np.zeros(self.n_env, dtype=np.float32)
        dones = np.zeros(self.n_env, dtype=np.bool_)
        infos: list[dict] = [{} for _ in range(self.n_env)]

        clipped_actions = np.clip(actions.astype(np.int64), 0, self.d - 1)

        for env_index in range(self.n_env):
            current_node = int(self.current_nodes[env_index])
            target_index = int(self.target_indices[env_index])
            target_node = int(self.target_nodes[target_index])

            dist_old = int(self.dist_pool[target_index, current_node])

            action = int(clipped_actions[env_index])
            next_node = int(self.out_neighbors[current_node, action])
            self.current_nodes[env_index] = next_node
            self.steps[env_index] += 1

            dist_new = int(self.dist_pool[target_index, next_node])
            success = next_node == target_node
            timeout = int(self.steps[env_index]) >= int(self.horizons[env_index])
            episode_done = bool(success or timeout)

            if self.sensing == "S0":
                old_effective = dist_old if dist_old < INF_DISTANCE else self.unreachable_penalty_dist
                new_effective = dist_new if dist_new < INF_DISTANCE else self.unreachable_penalty_dist
                progress = float(old_effective - new_effective)
                reward = float(np.clip(progress, -1.0, 1.0))
                if success:
                    reward += 1.0
            else:
                reward = 1.0 if success else self.s1_step_penalty
            if self.reward_noise_std > 0.0:
                reward += float(self.rng.normal(loc=0.0, scale=self.reward_noise_std))

            rewards[env_index] = reward

            if episode_done:
                dones[env_index] = True
                self.completed_episodes += 1
                episode_len = int(self.steps[env_index])
                shortest_dist = int(self.initial_dist[env_index])
                info = {
                    "episode_done": True,
                    "success": bool(success),
                    "episode_len": episode_len,
                    "shortest_dist": shortest_dist,
                    "target_index": target_index,
                }
                if success and shortest_dist > 0:
                    info["stretch"] = float(episode_len / shortest_dist)
                infos[env_index] = info

                self._reset_env(env_index)

        next_obs = self.get_obs()
        return next_obs, rewards, dones, infos
