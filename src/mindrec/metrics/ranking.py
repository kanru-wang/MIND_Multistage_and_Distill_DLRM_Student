from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def dcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order=np.argsort(-scores)[:k]
    gains=labels[order]
    discounts=1.0 / np.log2(np.arange(2, k + 2))
    return float((gains * discounts).sum())


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    ideal=np.sort(labels)[::-1]
    idcg=dcg_at_k(ideal, ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(labels, scores, k) / idcg


def mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    order=np.argsort(-scores)
    for rank,idx in enumerate(order, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order=np.argsort(-scores)[:k]
    hits=0
    s=0.0
    for i,idx in enumerate(order, start=1):
        if labels[idx] == 1:
            hits += 1
            s += hits / i
    denom=min(int(labels.sum()), k)
    return float(s / denom) if denom > 0 else 0.0


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order=np.argsort(-scores)[:k]
    hit=int(labels[order].sum())
    denom=int(labels.sum())
    return float(hit / denom) if denom > 0 else 0.0


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def ndcg_from_order(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    order=order[:k]
    gains=labels[order]
    discounts=1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg=float((gains * discounts).sum())
    ideal=np.sort(labels)[::-1]
    idcg=dcg_at_k(ideal, ideal, k)
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_from_order(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    order=order[:k]
    hit=int(labels[order].sum())
    denom=int(labels.sum())
    return float(hit / denom) if denom > 0 else 0.0
