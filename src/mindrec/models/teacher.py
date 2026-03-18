from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryAttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] True for valid
        q = self.query.expand(x.size(0), 1, x.size(2))
        key_padding_mask = ~mask  # True for pad positions
        out, _ = self.attn(
            q, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        return out.squeeze(1)


class TeacherTwoTower(nn.Module):
    def __init__(self, item_dim: int, hidden_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.item_proj = nn.Linear(item_dim, hidden_dim, bias=False)
        if item_dim == hidden_dim:
            nn.init.eye_(self.item_proj.weight)
        else:
            nn.init.xavier_uniform_(self.item_proj.weight)
        self.user_pool = HistoryAttentionPool(dim=hidden_dim, heads=heads)

    def encode_items(self, item_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_proj(item_emb), dim=-1)

    def encode_user(self, history_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hist_z = self.encode_items(history_emb)
        user_z = self.user_pool(hist_z, mask)
        return F.normalize(user_z, dim=-1)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return (a * b).sum(axis=-1)
