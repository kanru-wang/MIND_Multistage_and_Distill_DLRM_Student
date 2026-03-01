from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryAttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int=4) -> None:
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.query=nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] True for valid
        q=self.query.expand(x.size(0), 1, x.size(2))
        key_padding_mask=~mask  # True for pad positions
        out,_=self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return out.squeeze(1)


def l2_normalize(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    n=np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a=l2_normalize(a)
    b=l2_normalize(b)
    return (a * b).sum(axis=-1)
