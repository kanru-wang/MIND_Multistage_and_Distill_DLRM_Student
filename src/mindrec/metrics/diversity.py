from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def entropy(items: list[int]) -> float:
    if not items:
        return 0.0
    c=Counter(items)
    p=np.array([v / len(items) for v in c.values()], dtype=np.float32)
    return float(-(p * np.log(p + 1e-12)).sum())


def category_coverage(items: list[int]) -> int:
    return len(set(items))


def ild_from_similarity(sim_mat: np.ndarray) -> float:
    # sim_mat: [K,K] similarity; ILD = 1 - mean_{i<j} sim(i,j)
    k=sim_mat.shape[0]
    if k <= 1:
        return 0.0
    vals=[]
    for i in range(k):
        for j in range(i + 1, k):
            vals.append(float(sim_mat[i, j]))
    return float(1.0 - (np.mean(vals) if vals else 0.0))


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    inter=len(a & b)
    union=len(a | b)
    return float(inter / union) if union > 0 else 0.0
