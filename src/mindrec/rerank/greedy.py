from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from mindrec.metrics.diversity import jaccard
from mindrec.metrics.fairness import (
    catalog_target,
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


def _linear_exposure_from_counts(
    counts: dict[int, int], pos_sums: dict[int, int], k: int
) -> dict[int, float]:
    if k <= 0:
        return {}
    return {
        gid: float(count) - (float(pos_sums.get(gid, 0)) / float(k))
        for gid, count in counts.items()
    }


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
    pool_index = {nid: i for i, nid in enumerate(pool)}
    pool_scores = cand_scores[order]
    pool_is_new = [int(cand_is_new[i]) for i in order]
    pool_cats = [news_meta.get(nid, NewsMeta(0, 0, set())).cat_idx for nid in pool]

    # Precompute similarity for novelty
    sim_mat = None
    if novelty_sim == "teacher_cosine":
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
    chosen_exp_by_cat: dict[int, float] = {}
    chosen_cat_counts: Counter[int] = Counter()
    chosen_cat_pos_sums: dict[int, int] = {}
    chosen_new_count = 0
    chosen_new_pos_sum = 0
    chosen_new_exp = 0.0
    max_new_ent = int(coverage_cfg.get("max_new_entities_per_item", 3))
    cat_bonus = float(coverage_cfg.get("category_bonus", 1.0))
    ent_bonus = float(coverage_cfg.get("entity_bonus", 0.3))
    pos_mode = fairness_cfg.get("position_bias", "log")
    weights_by_len = {
        k: position_bias_weights(k, mode=pos_mode) for k in range(1, k_out + 1)
    }
    total_exp_by_len = {
        k: float(weights.sum()) for k, weights in weights_by_len.items()
    }

    target_mode = fairness_cfg.get("category_target", "catalog")
    if target_mode == "uniform":
        target_dist = normalize_dist(uniform_target([c for c in pool_cats if c != 0]))
    else:
        target_dist = normalize_dist(catalog_target([c for c in pool_cats if c != 0]))

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

    def fairness_penalty(i: int) -> float:
        if not fairness_cfg.get("enabled", False):
            return 0.0
        k = len(chosen) + 1
        cat_i = pool_cats[i]
        if pos_mode == "linear":
            trial_counts = dict(chosen_cat_counts)
            if cat_i != 0:
                trial_counts[cat_i] = trial_counts.get(cat_i, 0) + 1
            trial_pos_sums = dict(chosen_cat_pos_sums)
            if cat_i != 0:
                trial_pos_sums[cat_i] = trial_pos_sums.get(cat_i, 0) + len(chosen)
            exp = normalize_dist(
                _linear_exposure_from_counts(trial_counts, trial_pos_sums, k)
            )
            trial_new_count = chosen_new_count + int(pool_is_new[i] == 1)
            trial_new_pos_sum = chosen_new_pos_sum + (
                len(chosen) if int(pool_is_new[i]) == 1 else 0
            )
            new_exp = float(trial_new_count) - (float(trial_new_pos_sum) / float(k))
        else:
            next_w = float(weights_by_len[k][-1])
            trial_exp = dict(chosen_exp_by_cat)
            if cat_i != 0:
                trial_exp[cat_i] = trial_exp.get(cat_i, 0.0) + next_w
            exp = normalize_dist(trial_exp)
            new_exp = chosen_new_exp + (next_w if int(pool_is_new[i]) == 1 else 0.0)

        kl = kl_divergence(exp, target_dist)
        l1 = l1_distance(exp, target_dist)
        pen = 0.5 * kl + 0.5 * l1

        # New item floor
        floor = float(fairness_cfg.get("new_item_floor", 0.0))
        if floor > 0.0:
            total_exp = total_exp_by_len[k]
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

            if fairness_cfg.get("enabled", False):
                val -= float(fairness_cfg.get("penalty_weight", 0.5)) * fairness_penalty(i)

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
            chosen_cat_counts[m.cat_idx] += 1
            chosen_cat_pos_sums[m.cat_idx] = chosen_cat_pos_sums.get(m.cat_idx, 0) + (
                len(chosen) - 1
            )
        chosen_ents |= m.ent
        if pos_mode == "linear":
            if int(pool_is_new[pool_index[best]]) == 1:
                chosen_new_count += 1
                chosen_new_pos_sum += len(chosen) - 1
        else:
            pos_w = float(weights_by_len[len(chosen)][-1])
            if m.cat_idx != 0:
                chosen_exp_by_cat[m.cat_idx] = chosen_exp_by_cat.get(m.cat_idx, 0.0) + pos_w
            if int(pool_is_new[pool_index[best]]) == 1:
                chosen_new_exp += pos_w

    return {
        "ranked_news_id": chosen,
    }
