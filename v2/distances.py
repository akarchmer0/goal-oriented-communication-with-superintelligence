from collections import deque

import numpy as np
from tqdm import tqdm


INF_DISTANCE = np.int32(1_000_000_000)


def bfs_distances_to_target(
    rev_neighbors: list[np.ndarray],
    target: int,
    n: int,
    inf_distance: np.int32 = INF_DISTANCE,
) -> np.ndarray:
    dist = np.full(n, inf_distance, dtype=np.int32)
    queue: deque[int] = deque()
    dist[target] = 0
    queue.append(target)

    while queue:
        node = queue.popleft()
        next_dist = int(dist[node]) + 1
        for parent in rev_neighbors[node]:
            parent_idx = int(parent)
            if dist[parent_idx] == inf_distance:
                dist[parent_idx] = next_dist
                queue.append(parent_idx)
    return dist


def precompute_distance_pool(
    rev_neighbors: list[np.ndarray],
    n: int,
    t_pool_size: int,
    seed: int,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t_pool_size = min(t_pool_size, n)
    target_nodes = rng.choice(n, size=t_pool_size, replace=False).astype(np.int32)

    dist_pool = np.empty((t_pool_size, n), dtype=np.int32)
    iterator = range(t_pool_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Precomputing BFS distances", leave=False)

    for target_index in iterator:
        target = int(target_nodes[target_index])
        dist_pool[target_index] = bfs_distances_to_target(rev_neighbors, target, n)

    return target_nodes, dist_pool


def sample_reachable_source(
    dist_to_target: np.ndarray,
    rng: np.random.Generator,
    inf_distance: np.int32 = INF_DISTANCE,
    max_attempts: int = 10_000,
) -> int:
    n = dist_to_target.shape[0]
    for _ in range(max_attempts):
        source = int(rng.integers(0, n))
        if 0 < int(dist_to_target[source]) < int(inf_distance):
            return source

    reachable = np.where((dist_to_target > 0) & (dist_to_target < inf_distance))[0]
    if reachable.size == 0:
        raise RuntimeError("No reachable source nodes found for selected target")
    return int(rng.choice(reachable))

