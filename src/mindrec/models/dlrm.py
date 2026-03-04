from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(
    in_dim: int,
    layer_dims: list[int],
    dropout: float = 0.0,
    last_activation: bool = True,
) -> nn.Sequential:
    layers = []
    d = in_dim
    for i, od in enumerate(layer_dims):
        layers.append(nn.Linear(d, od))
        if i < len(layer_dims) - 1 or last_activation:
            layers.append(nn.ReLU())
        if dropout > 0.0 and i < len(layer_dims) - 1:
            layers.append(nn.Dropout(dropout))
        d = od
    return nn.Sequential(*layers)


class AttentionFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.out = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [B,D], kv: [B,T,D]
        q2 = self.q_proj(q).unsqueeze(1)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        out, _ = self.attn(q2, k, v, need_weights=False)
        return self.out(out.squeeze(1))


class DLRMStudent(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_news: int,
        n_cats: int,
        n_subcats: int,
        dense_dim: int,
        emb_dim: int = 64,
        bottom_mlp: list[int] | None = None,
        top_mlp: list[int] | None = None,
        dropout: float = 0.0,
        teacher_dim: int | None = None,
        fusion_heads: int = 4,
    ) -> None:
        super().__init__()
        bottom_mlp = bottom_mlp or [128, 64]
        top_mlp = top_mlp or [256, 128, 1]

        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.news_emb = nn.Embedding(n_news, emb_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, emb_dim, padding_idx=0)
        self.subcat_emb = nn.Embedding(n_subcats, emb_dim, padding_idx=0)

        self.bottom = make_mlp(
            dense_dim, bottom_mlp, dropout=dropout, last_activation=True
        )
        d_bottom = bottom_mlp[-1]

        self.use_teacher = teacher_dim is not None
        self.teacher_dim = teacher_dim

        if self.use_teacher:
            self.teacher_u_proj = nn.Linear(teacher_dim, emb_dim)
            self.teacher_i_proj = nn.Linear(teacher_dim, emb_dim)
            self.fusion = AttentionFusion(dim=emb_dim, heads=fusion_heads)

        # DLRM interaction dims:
        # features: dense_bottom + 4 embeddings (+ optional fusion vector)
        self.n_feat = 1 + 4 + (1 if self.use_teacher else 0)
        n_inter = self.n_feat * (self.n_feat - 1) // 2
        top_in = d_bottom + 4 * emb_dim + (emb_dim if self.use_teacher else 0) + n_inter

        self.top = make_mlp(top_in, top_mlp, dropout=dropout, last_activation=False)

    def forward(
        self,
        user_idx: torch.Tensor,
        news_idx: torch.Tensor,
        cat_idx: torch.Tensor,
        subcat_idx: torch.Tensor,
        dense: torch.Tensor,
        teacher_user_emb: Optional[torch.Tensor] = None,
        teacher_item_emb: Optional[torch.Tensor] = None,
        return_repr: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Embeddings
        eu = self.user_emb(user_idx)
        ei = self.news_emb(news_idx)
        ec = self.cat_emb(cat_idx)
        es = self.subcat_emb(subcat_idx)

        xd = self.bottom(dense)

        extras = []
        if self.use_teacher:
            if teacher_user_emb is None or teacher_item_emb is None:
                raise ValueError("Teacher embeddings required when teacher_dim is set.")
            tu = self.teacher_u_proj(teacher_user_emb)
            ti = self.teacher_i_proj(teacher_item_emb)
            kv = torch.stack([tu, ti], dim=1)  # [B,2,D]
            q = eu + xd  # query from user emb + dense summary
            zf = self.fusion(q=q, kv=kv)
            extras.append(zf)

        # Build feature list for interactions (project dense into emb_dim by repeating)
        # We interact: [xd_proj, eu, ei, ec, es, (zf)]
        # xd has d_bottom, not emb_dim; use a linear projection to emb_dim via a simple padding/truncation trick
        # but keep the raw xd for top-mlp too.
        # To keep it simple, create xd_emb with a learned linear mapping:
        if not hasattr(self, "xd_proj"):
            self.xd_proj = nn.Linear(xd.size(1), eu.size(1)).to(xd.device)
        xd_emb = self.xd_proj(xd)

        feats = [xd_emb, eu, ei, ec, es] + extras
        inter = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                inter.append((feats[i] * feats[j]).sum(dim=1, keepdim=True))
        inter_vec = (
            torch.cat(inter, dim=1)
            if inter
            else torch.zeros((xd.size(0), 0), device=xd.device)
        )

        concat = torch.cat([xd, eu, ei, ec, es] + extras + [inter_vec], dim=1)
        logit = self.top(concat).squeeze(1)

        rep = None
        if return_repr:
            # Representation used for distillation alignment
            rep = torch.cat([eu, ei] + extras, dim=1)
        return logit, rep
