from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from mindrec.metrics.diversity import jaccard
from mindrec.metrics.fairness import (
    catalog_target,
    exposure_from_ranking,
    gini,
    kl_divergence,
    l1_distance,
    normalize_dist,
    uniform_target,
)
from mindrec.utils import position_bias_weights


def _parse_entities(s: str) -> set[int]:
    # MIND entities are JSON-like; keep robust.
    if not isinstance(s, str) or not s.strip():
        return set()
    try:
        data = json.loads(s)
    except Exception:
        return set()
    out = set()
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                name = str(e.get("Label") or e.get("WikidataId") or e.get("Type") or "")
                if name:
                    out.add(abs(hash(name)) % (2**31 - 1))
    return out


@dataclass
class NewsMeta:
    cat_idx: int
    subcat_idx: int
    ent: set[int]


def build_news_meta(news_df) -> dict[str, NewsMeta]:
    # key by news_id
    meta = {}
    for _, r in news_df.iterrows():
        ent = _parse_entities(r.get("title_entities", "")) | _parse_entities(
            r.get("abstract_entities", "")
        )
        meta[str(r["news_id"])] = NewsMeta(
            cat_idx=int(r.get("cat_idx", 0)),
            subcat_idx=int(r.get("subcat_idx", 0)),
            ent=ent,
        )
    return meta


def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    # x: [K,D] assumed normalized
    return x @ x.T


def greedy_rerank(
    cand_news_id: list[str],
    cand_scores: np.ndarray,
    cand_is_new: list[int],
    news_meta: dict[str, NewsMeta],
    item_teacher_emb: np.ndarray,
    k_out: int,
    pool_size: int,
    relevance_weight: float,
    novelty_weight: float,
    coverage_weight: float,
    novelty_sim: str,
    coverage_cfg: dict[str, Any],
    fairness_cfg: dict[str, Any],
) -> dict[str, Any]:
    # Work on top pool_size by relevance
    order = np.argsort(-cand_scores)[:pool_size]
    pool = [cand_news_id[i] for i in order]
    pool_scores = cand_scores[order]
    pool_is_new = [int(cand_is_new[i]) for i in order]

    # Precompute similarity for novelty
    sim_mat = None
    if novelty_sim == "teacher_cosine":
        idx = [
            int(news_meta.get(nid, NewsMeta(0, 0, set())).cat_idx) for nid in pool
        ]  # placeholder
        # Use item teacher embeddings by news_idx: we don't have news_idx here; approximate by hashing news_id? Not ok.
        # In this repo, we store teacher embeddings by news_idx. So we also expect caller to pass emb vectors for pool directly.
        # Here we'll accept item_teacher_emb already as [pool_size,D] embeddings.
        x = item_teacher_emb.astype(np.float32)
        # Normalize just in case
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        sim_mat = cosine_sim_matrix(x)
    elif novelty_sim in ["entity_jaccard", "category"]:
        sim_mat = None
    else:
        raise ValueError(f"Unknown novelty_sim: {novelty_sim}")

    chosen = []
    chosen_idx = []
    chosen_set = set()
    chosen_cats = set()
    chosen_ents = set()
    max_new_ent = int(coverage_cfg.get("max_new_entities_per_item", 3))
    cat_bonus = float(coverage_cfg.get("category_bonus", 1.0))
    ent_bonus = float(coverage_cfg.get("entity_bonus", 0.3))

    def novelty(i: int) -> float:
        if not chosen_idx:
            return 0.0
        if novelty_sim == "teacher_cosine" and sim_mat is not None:
            sims = [float(sim_mat[i, j]) for j in chosen_idx]
            return -max(sims)
        if novelty_sim == "category":
            ci = news_meta.get(pool[i], NewsMeta(0, 0, set())).cat_idx
            sims = [
                (
                    1.0
                    if ci == news_meta.get(pool[j], NewsMeta(0, 0, set())).cat_idx
                    else 0.0
                )
                for j in chosen_idx
            ]
            return -max(sims)
        # entity_jaccard
        ei = news_meta.get(pool[i], NewsMeta(0, 0, set())).ent
        sims = [
            jaccard(ei, news_meta.get(pool[j], NewsMeta(0, 0, set())).ent)
            for j in chosen_idx
        ]
        return -max(sims)

    def coverage(i: int) -> float:
        m = news_meta.get(pool[i], NewsMeta(0, 0, set()))
        bonus = 0.0
        if m.cat_idx not in chosen_cats and m.cat_idx != 0:
            bonus += cat_bonus
        if m.ent:
            new_ents = list(m.ent - chosen_ents)
            bonus += ent_bonus * float(min(len(new_ents), max_new_ent))
        return bonus

    def fairness_penalty(ranked_ids: list[str]) -> float:
        if not fairness_cfg.get("enabled", False):
            return 0.0
        k = len(ranked_ids)
        w = position_bias_weights(k, mode=fairness_cfg.get("position_bias", "log"))
        cats = [news_meta.get(nid, NewsMeta(0, 0, set())).cat_idx for nid in ranked_ids]
        exp = normalize_dist(exposure_from_ranking(cats, w))

        target_mode = fairness_cfg.get("category_target", "catalog")
        if target_mode == "uniform":
            tgt = uniform_target(cats)
        else:
            tgt = catalog_target(cats)
        tgt = normalize_dist(tgt)
        kl = kl_divergence(exp, tgt)
        l1 = l1_distance(exp, tgt)
        pen = 0.5 * kl + 0.5 * l1

        # New item floor
        floor = float(fairness_cfg.get("new_item_floor", 0.0))
        if floor > 0.0:
            is_new = [
                int(pool_is_new[pool.index(nid)]) if nid in pool else 0
                for nid in ranked_ids
            ]
            new_exp = float(
                sum(wi for wi, flag in zip(w.tolist(), is_new) if flag == 1)
            )
            total_exp = float(sum(w.tolist()))
            frac = (new_exp / total_exp) if total_exp > 0 else 0.0
            if frac < floor:
                pen += (floor - frac) * 2.0
        return float(pen)

    # Greedy selection
    for _ in range(min(k_out, len(pool))):
        best = None
        best_i = None
        best_val = -1e18
        for i, nid in enumerate(pool):
            if nid in chosen_set:
                continue
            rel = float(pool_scores[i])
            val = (
                relevance_weight * rel
                + novelty_weight * novelty(i)
                + coverage_weight * coverage(i)
            )

            if fairness_cfg.get("enabled", False) and chosen:
                trial = chosen + [nid]
                val -= float(
                    fairness_cfg.get("penalty_weight", 0.5)
                ) * fairness_penalty(trial)

            if val > best_val:
                best_val = val
                best = nid
                best_i = i

        if best is None:
            break
        chosen.append(best)
        chosen_idx.append(int(best_i))
        chosen_set.add(best)
        m = news_meta.get(best, NewsMeta(0, 0, set()))
        if m.cat_idx != 0:
            chosen_cats.add(m.cat_idx)
        chosen_ents |= m.ent

    return {
        "ranked_news_id": chosen,
    }
