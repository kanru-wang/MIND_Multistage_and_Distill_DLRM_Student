from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np


def exposure_from_ranking(
    group_ids: list[int], position_weights: np.ndarray
) -> dict[int, float]:
    exp = {}
    for gid, w in zip(group_ids, position_weights.tolist()):
        exp[gid] = exp.get(gid, 0.0) + float(w)
    return exp


def normalize_dist(d: dict[int, float]) -> dict[int, float]:
    s = sum(d.values())
    if s <= 0:
        return {k: 0.0 for k in d}
    return {k: v / s for k, v in d.items()}


def kl_divergence(
    p: dict[int, float], q: dict[int, float], eps: float = 1e-12
) -> float:
    # KL(p||q)
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        pk = float(p.get(k, 0.0))
        qk = float(q.get(k, 0.0))
        if pk <= 0:
            continue
        s += pk * np.log((pk + eps) / (qk + eps))
    return float(s)


def l1_distance(p: dict[int, float], q: dict[int, float]) -> float:
    keys = set(p) | set(q)
    return float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def gini(values: list[float]) -> float:
    x = np.array(values, dtype=np.float32)
    if x.size == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    if cum[-1] <= 0:
        return 0.0
    g = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(g)


def catalog_target(group_ids: list[int]) -> dict[int, float]:
    c = Counter(group_ids)
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} if total > 0 else {}


def uniform_target(group_ids: list[int]) -> dict[int, float]:
    keys = sorted(set(group_ids))
    if not keys:
        return {}
    p = 1.0 / len(keys)
    return {k: p for k in keys}
