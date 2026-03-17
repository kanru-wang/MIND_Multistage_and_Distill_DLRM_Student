from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from mindrec.metrics.diversity import jaccard
from mindrec.metrics.fairness import (
    catalog_target,
    normalize_dist,
    uniform_target,
)
from mindrec.utils import position_bias_weights


def _stable_entity_id(name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


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
                    out.add(_stable_entity_id(name))
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


def _build_novelty_similarity(
    novelty_sim: str,
    pool: list[str],
    news_meta: dict[str, NewsMeta],
    item_teacher_emb: np.ndarray,
) -> np.ndarray | None:
    if novelty_sim == "teacher_cosine":
        x = item_teacher_emb.astype(np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return cosine_sim_matrix(x)
    if novelty_sim == "category":
        cats = np.array(
            [news_meta.get(nid, NewsMeta(0, 0, set())).cat_idx for nid in pool],
            dtype=np.int64,
        )
        sim = (cats[:, None] == cats[None, :]).astype(np.float32)
        # Treat unknown category 0 as missing signal rather than a real shared category.
        unknown_mask = cats == 0
        sim[unknown_mask, :] = 0.0
        sim[:, unknown_mask] = 0.0
        np.fill_diagonal(sim, 1.0)
        return sim
    if novelty_sim == "entity_jaccard":
        n = len(pool)
        sim = np.eye(n, dtype=np.float32)
        ents = [news_meta.get(nid, NewsMeta(0, 0, set())).ent for nid in pool]
        for i in range(n):
            for j in range(i + 1, n):
                score = float(jaccard(ents[i], ents[j]))
                sim[i, j] = score
                sim[j, i] = score
        return sim
    raise ValueError(f"Unknown novelty_sim: {novelty_sim}")


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
    sim_mat = _build_novelty_similarity(
        novelty_sim=novelty_sim,
        pool=pool,
        news_meta=news_meta,
        item_teacher_emb=item_teacher_emb,
    )

    chosen = []
    chosen_idx = []
    chosen_set = set()
    chosen_cats = set()
    chosen_ents = set()
    max_sim_to_chosen = np.zeros(len(pool), dtype=np.float32)
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
    target_keys = list(target_dist.keys())

    def novelty(i: int) -> float:
        if not chosen_idx:
            return 0.0
        return -float(max_sim_to_chosen[i])

    def coverage(i: int) -> float:
        m = news_meta.get(pool[i], NewsMeta(0, 0, set()))
        bonus = 0.0
        if m.cat_idx not in chosen_cats and m.cat_idx != 0:
            bonus += cat_bonus
        if m.ent:
            new_ents = list(m.ent - chosen_ents)
            bonus += ent_bonus * float(min(len(new_ents), max_new_ent))
        return bonus

    def fairness_penalty_log(cat_i: int, is_new_i: int, k: int) -> float:
        next_w = float(weights_by_len[k][-1])
        total_exp = total_exp_by_len[k]
        kl = 0.0
        l1 = 0.0
        for gid in target_keys:
            raw_exp = chosen_exp_by_cat.get(gid, 0.0)
            if gid == cat_i and gid != 0:
                raw_exp += next_w
            pk = (raw_exp / total_exp) if total_exp > 0 else 0.0
            qk = float(target_dist.get(gid, 0.0))
            if pk > 0.0:
                kl += pk * np.log((pk + 1e-12) / (qk + 1e-12))
            l1 += abs(pk - qk)

        pen = 0.5 * float(kl) + 0.5 * float(l1)
        floor = float(fairness_cfg.get("new_item_floor", 0.0))
        if floor > 0.0:
            new_exp = chosen_new_exp + (next_w if is_new_i == 1 else 0.0)
            frac = (new_exp / total_exp) if total_exp > 0 else 0.0
            if frac < floor:
                pen += (floor - frac) * 2.0
        return pen

    def fairness_penalty_linear(cat_i: int, is_new_i: int, k: int) -> float:
        total_exp = total_exp_by_len[k]
        kl = 0.0
        l1 = 0.0
        for gid in target_keys:
            count = chosen_cat_counts.get(gid, 0)
            pos_sum = chosen_cat_pos_sums.get(gid, 0)
            if gid == cat_i and gid != 0:
                count += 1
                pos_sum += len(chosen)
            raw_exp = float(count) - (float(pos_sum) / float(k))
            pk = (raw_exp / total_exp) if total_exp > 0 else 0.0
            qk = float(target_dist.get(gid, 0.0))
            if pk > 0.0:
                kl += pk * np.log((pk + 1e-12) / (qk + 1e-12))
            l1 += abs(pk - qk)

        pen = 0.5 * float(kl) + 0.5 * float(l1)
        floor = float(fairness_cfg.get("new_item_floor", 0.0))
        if floor > 0.0:
            new_count = chosen_new_count + int(is_new_i == 1)
            new_pos_sum = chosen_new_pos_sum + (len(chosen) if is_new_i == 1 else 0)
            new_exp = float(new_count) - (float(new_pos_sum) / float(k))
            frac = (new_exp / total_exp) if total_exp > 0 else 0.0
            if frac < floor:
                pen += (floor - frac) * 2.0
        return pen

    def fairness_penalty(i: int) -> float:
        if not fairness_cfg.get("enabled", False):
            return 0.0
        k = len(chosen) + 1
        cat_i = pool_cats[i]
        is_new_i = int(pool_is_new[i])
        if pos_mode == "linear":
            return float(fairness_penalty_linear(cat_i, is_new_i, k))
        return float(fairness_penalty_log(cat_i, is_new_i, k))

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
        if sim_mat is not None and best_i is not None:
            max_sim_to_chosen = np.maximum(max_sim_to_chosen, sim_mat[:, int(best_i)])
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
