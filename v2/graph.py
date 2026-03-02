from dataclasses import dataclass

import numpy as np


@dataclass
class DirectedRegularGraph:
    n: int
    d: int
    out_neighbors: np.ndarray
    rev_neighbors: list[np.ndarray]


def build_directed_regular_graph(
    n: int,
    d: int,
    seed: int,
    allow_self_loops: bool = False,
) -> DirectedRegularGraph:
    if d <= 0:
        raise ValueError("d must be positive")
    if not allow_self_loops and d >= n:
        raise ValueError("d must be < n when self loops are disabled")

    rng = np.random.default_rng(seed)
    out_neighbors = np.empty((n, d), dtype=np.int32)

    for node in range(n):
        selected_set: set[int] = set()
        selected_ordered: list[int] = []
        while len(selected_set) < d:
            candidate = int(rng.integers(0, n))
            if not allow_self_loops and candidate == node:
                continue
            if candidate in selected_set:
                continue
            selected_set.add(candidate)
            selected_ordered.append(candidate)
        out_neighbors[node] = np.asarray(selected_ordered, dtype=np.int32)

    rev_lists: list[list[int]] = [[] for _ in range(n)]
    for src in range(n):
        for dst in out_neighbors[src]:
            rev_lists[int(dst)].append(src)

    rev_neighbors = [np.asarray(parents, dtype=np.int32) for parents in rev_lists]
    return DirectedRegularGraph(
        n=n,
        d=d,
        out_neighbors=out_neighbors,
        rev_neighbors=rev_neighbors,
    )
