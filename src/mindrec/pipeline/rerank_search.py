from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.metrics.diversity import category_coverage, entropy, ild_from_similarity
from mindrec.metrics.fairness import (
    catalog_target,
    exposure_from_ranking,
    gini,
    kl_divergence,
    normalize_dist,
    uniform_target,
)
from mindrec.metrics.ranking import ndcg_at_k, ndcg_from_order, recall_at_k, recall_from_order
from mindrec.pipeline.evaluate import _load_model
from mindrec.rerank.greedy import build_news_meta, cosine_sim_matrix, greedy_rerank
from mindrec.utils import position_bias_weights, save_json


@dataclass
class ImpressionScores:
    labels: np.ndarray
    cand_news_id: list[str]
    cand_news_idx: np.ndarray
    cand_is_new: list[int]
    scores: np.ndarray


def _cat_idx(news_meta: dict[str, Any], news_id: str) -> int:
    meta = news_meta.get(news_id)
    return int(meta.cat_idx) if meta is not None else 0


def _category_reference(
    cand_news_id: list[str], news_meta: dict[str, Any]
) -> list[int]:
    return [
        cat_idx
        for cat_idx in (_cat_idx(news_meta, nid) for nid in cand_news_id)
        if cat_idx != 0
    ]


def _new_item_exposure_frac(
    weights: np.ndarray, ranking_idx: list[int], cand_is_new: list[int]
) -> float:
    new_exp = sum(
        float(weight)
        for weight, idx in zip(weights.tolist(), ranking_idx)
        if int(cand_is_new[idx]) == 1
    )
    return float(new_exp / (float(weights.sum()) + 1e-12))


def _resolve_device(cfg: dict[str, Any]) -> torch.device:
    device_str = cfg["ranker"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def _score_impressions(
    cfg: dict[str, Any],
    proc_root: Path,
    runs_root: Path,
    device: torch.device,
) -> tuple[list[ImpressionScores], dict[str, Any]]:
    model, teacher_user, teacher_item = _load_model(cfg, proc_root, runs_root, device)
    impr = pd.read_parquet(proc_root / "dev_impressions.parquet")

    scored: list[ImpressionScores] = []
    with torch.no_grad():
        for _, r in tqdm(impr.iterrows(), total=len(impr), desc="Score impressions"):
            labels = np.array(r["cand_label"], dtype=np.int32)
            if labels.sum() <= 0:
                continue

            user_idx = int(r["user_idx"])
            cand_news_id = list(r["cand_news_id"])
            cand_news_idx = np.array(r["cand_news_idx"], dtype=np.int64)
            cand_cat_idx = np.array(r["cand_cat_idx"], dtype=np.int64)
            cand_subcat_idx = np.array(r["cand_subcat_idx"], dtype=np.int64)
            cand_is_new = [int(x) for x in list(r["cand_is_new_item"])]
            cand_clicks_log1p = np.array(r["cand_item_clicks_log1p"], dtype=np.float32)
            hlen = float(r["history_len"])
            dense = np.stack(
                [np.full_like(cand_clicks_log1p, hlen), cand_clicks_log1p], axis=1
            )

            tu = torch.tensor(
                teacher_user[user_idx : user_idx + 1],
                dtype=torch.float32,
                device=device,
            ).repeat(len(cand_news_idx), 1)
            ti = torch.tensor(
                teacher_item[cand_news_idx], dtype=torch.float32, device=device
            )

            logits = []
            bs = 2048
            for i in range(0, len(cand_news_idx), bs):
                sl = slice(i, i + bs)
                b_user = torch.tensor(
                    [user_idx] * len(cand_news_idx[sl]), dtype=torch.long, device=device
                )
                b_news = torch.tensor(
                    cand_news_idx[sl], dtype=torch.long, device=device
                )
                b_cat = torch.tensor(cand_cat_idx[sl], dtype=torch.long, device=device)
                b_sub = torch.tensor(
                    cand_subcat_idx[sl], dtype=torch.long, device=device
                )
                b_dense = torch.tensor(dense[sl], dtype=torch.float32, device=device)
                logit, _ = model(
                    user_idx=b_user,
                    news_idx=b_news,
                    cat_idx=b_cat,
                    subcat_idx=b_sub,
                    dense=b_dense,
                    teacher_user_emb=tu[sl],
                    teacher_item_emb=ti[sl],
                )
                logits.append(logit.detach().cpu().numpy())

            scored.append(
                ImpressionScores(
                    labels=labels,
                    cand_news_id=cand_news_id,
                    cand_news_idx=cand_news_idx,
                    cand_is_new=cand_is_new,
                    scores=np.concatenate(logits, axis=0),
                )
            )

    return scored, {"teacher_item": teacher_item}


def _evaluate_baseline(
    scored_impressions: list[ImpressionScores],
    teacher_item: np.ndarray,
    news_meta: dict[str, Any],
    k_out: int,
    position_bias: str,
    category_target: str,
) -> dict[str, float]:
    base_ndcg = []
    base_recall = []
    base_ild = []
    base_cov = []
    base_ent = []
    base_fair_kl = []
    base_fair_gini = []
    base_new_exp = []
    w = position_bias_weights(k_out, mode=position_bias)

    for row in scored_impressions:
        base_order = np.argsort(-row.scores)[:k_out]
        base_ids = [row.cand_news_id[i] for i in base_order.tolist()]
        cand_cat_ref = _category_reference(row.cand_news_id, news_meta)
        base_cats = [_cat_idx(news_meta, nid) for nid in base_ids]

        base_ndcg.append(ndcg_at_k(row.labels, row.scores, k_out))
        base_recall.append(recall_at_k(row.labels, row.scores, k_out))
        base_cov.append(category_coverage([c for c in base_cats if c != 0]))
        base_ent.append(entropy([c for c in base_cats if c != 0]))

        base_emb = teacher_item[row.cand_news_idx[base_order]]
        base_emb = base_emb / (np.linalg.norm(base_emb, axis=1, keepdims=True) + 1e-12)
        base_ild.append(ild_from_similarity(cosine_sim_matrix(base_emb)))

        exp = normalize_dist(exposure_from_ranking(base_cats, w))
        tgt = (
            uniform_target(cand_cat_ref)
            if category_target == "uniform"
            else catalog_target(cand_cat_ref)
        )
        tgt = normalize_dist(tgt)
        base_fair_kl.append(kl_divergence(exp, tgt))
        base_fair_gini.append(gini(list(exp.values())))

        base_new_exp.append(
            _new_item_exposure_frac(w, base_order.tolist(), row.cand_is_new)
        )

    return {
        "ndcg@k": float(np.mean(base_ndcg) if base_ndcg else 0.0),
        "recall@k": float(np.mean(base_recall) if base_recall else 0.0),
        "ild": float(np.mean(base_ild) if base_ild else 0.0),
        "category_coverage": float(np.mean(base_cov) if base_cov else 0.0),
        "category_entropy": float(np.mean(base_ent) if base_ent else 0.0),
        "fairness_kl": float(np.mean(base_fair_kl) if base_fair_kl else 0.0),
        "fairness_gini": float(np.mean(base_fair_gini) if base_fair_gini else 0.0),
        "new_item_exposure_frac": float(np.mean(base_new_exp) if base_new_exp else 0.0),
    }


def _evaluate_candidate(
    scored_impressions: list[ImpressionScores],
    teacher_item: np.ndarray,
    news_meta: dict[str, Any],
    k_out: int,
    pool_size: int,
    position_bias: str,
    coverage_cfg: dict[str, Any],
    fairness_cfg: dict[str, Any],
    relevance_weight: float,
    novelty_weight: float,
    coverage_weight: float,
    novelty_sim: str,
) -> dict[str, Any]:
    rr_ndcg = []
    rr_recall = []
    rr_ild = []
    rr_cov = []
    rr_ent = []
    rr_fair_kl = []
    rr_fair_gini = []
    rr_new_exp = []
    w = position_bias_weights(k_out, mode=position_bias)

    for row in scored_impressions:
        pool_order = np.argsort(-row.scores)[:pool_size]
        pool_emb = teacher_item[row.cand_news_idx[pool_order]]
        cand_cat_ref = _category_reference(row.cand_news_id, news_meta)
        rr = greedy_rerank(
            cand_news_id=row.cand_news_id,
            cand_scores=row.scores,
            cand_is_new=row.cand_is_new,
            news_meta=news_meta,
            item_teacher_emb=pool_emb,
            k_out=k_out,
            pool_size=pool_size,
            relevance_weight=relevance_weight,
            novelty_weight=novelty_weight,
            coverage_weight=coverage_weight,
            novelty_sim=novelty_sim,
            coverage_cfg=coverage_cfg,
            fairness_cfg=fairness_cfg,
        )

        rr_ids = rr["ranked_news_id"]
        cand_idx_by_id = {nid: idx for idx, nid in enumerate(row.cand_news_id)}
        rr_idx = [cand_idx_by_id[nid] for nid in rr_ids]
        rr_cats = [_cat_idx(news_meta, nid) for nid in rr_ids]

        rr_ndcg.append(ndcg_from_order(row.labels, np.array(rr_idx), k_out))
        rr_recall.append(recall_from_order(row.labels, np.array(rr_idx), k_out))
        rr_cov.append(category_coverage([c for c in rr_cats if c != 0]))
        rr_ent.append(entropy([c for c in rr_cats if c != 0]))

        rr_emb = teacher_item[row.cand_news_idx[rr_idx]]
        rr_emb = rr_emb / (np.linalg.norm(rr_emb, axis=1, keepdims=True) + 1e-12)
        rr_ild.append(ild_from_similarity(cosine_sim_matrix(rr_emb)))

        exp = normalize_dist(exposure_from_ranking(rr_cats, w))
        tgt = (
            uniform_target(cand_cat_ref)
            if fairness_cfg.get("category_target", "catalog") == "uniform"
            else catalog_target(cand_cat_ref)
        )
        tgt = normalize_dist(tgt)
        rr_fair_kl.append(kl_divergence(exp, tgt))
        rr_fair_gini.append(gini(list(exp.values())))

        rr_new_exp.append(_new_item_exposure_frac(w, rr_idx, row.cand_is_new))

    metrics = {
        "ndcg@k": float(np.mean(rr_ndcg) if rr_ndcg else 0.0),
        "recall@k": float(np.mean(rr_recall) if rr_recall else 0.0),
        "ild": float(np.mean(rr_ild) if rr_ild else 0.0),
        "category_coverage": float(np.mean(rr_cov) if rr_cov else 0.0),
        "category_entropy": float(np.mean(rr_ent) if rr_ent else 0.0),
        "fairness_kl": float(np.mean(rr_fair_kl) if rr_fair_kl else 0.0),
        "fairness_gini": float(np.mean(rr_fair_gini) if rr_fair_gini else 0.0),
        "new_item_exposure_frac": float(np.mean(rr_new_exp) if rr_new_exp else 0.0),
    }
    metrics["weights"] = {
        "relevance": relevance_weight,
        "novelty": novelty_weight,
        "coverage": coverage_weight,
    }
    metrics["fairness"] = {
        "penalty_weight": float(fairness_cfg.get("penalty_weight", 0.0)),
        "new_item_floor": float(fairness_cfg.get("new_item_floor", 0.0)),
        "category_target": fairness_cfg.get("category_target", "catalog"),
    }
    metrics["novelty_sim"] = novelty_sim
    return metrics


def _make_constraint(baseline: dict[str, float]) -> dict[str, float]:
    return {
        "max_ndcg_drop_pct": 1.0,
        "min_new_item_exposure_gain": 0.015,
        "min_category_coverage_gain": 0.30,
        "max_fairness_kl_increase": 0.002,
        "baseline_ndcg@k": baseline["ndcg@k"],
        "baseline_new_item_exposure_frac": baseline["new_item_exposure_frac"],
        "baseline_category_coverage": baseline["category_coverage"],
        "baseline_fairness_kl": baseline["fairness_kl"],
    }


def _constraint_check(
    baseline: dict[str, float], metrics: dict[str, Any], constraint: dict[str, float]
) -> dict[str, Any]:
    ndcg_drop_pct = 100.0 * max(
        0.0, (baseline["ndcg@k"] - metrics["ndcg@k"]) / max(baseline["ndcg@k"], 1e-12)
    )
    new_gain = metrics["new_item_exposure_frac"] - baseline["new_item_exposure_frac"]
    cov_gain = metrics["category_coverage"] - baseline["category_coverage"]
    fair_kl_delta = metrics["fairness_kl"] - baseline["fairness_kl"]
    feasible = (
        ndcg_drop_pct <= constraint["max_ndcg_drop_pct"]
        and new_gain >= constraint["min_new_item_exposure_gain"]
        and cov_gain >= constraint["min_category_coverage_gain"]
        and fair_kl_delta <= constraint["max_fairness_kl_increase"]
    )
    return {
        "feasible": feasible,
        "ndcg_drop_pct": float(ndcg_drop_pct),
        "new_item_exposure_gain": float(new_gain),
        "category_coverage_gain": float(cov_gain),
        "fairness_kl_delta": float(fair_kl_delta),
    }


def run_rerank_search(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    out_root = ensure_dir(runs_root / "eval")
    rr_cfg = cfg["rerank"]

    device = _resolve_device(cfg)
    news = pd.read_parquet(proc_root / "news.parquet")
    news_meta = build_news_meta(news)
    scored_impressions, assets = _score_impressions(cfg, proc_root, runs_root, device)
    teacher_item = assets["teacher_item"]

    k_out = int(rr_cfg["k_out"])
    pool_size = int(rr_cfg["pool_size"])
    position_bias = rr_cfg.get("position_bias", "log")
    coverage_cfg = dict(rr_cfg.get("coverage", {}))
    fairness_base = dict(rr_cfg.get("fairness", {}))
    fairness_base["position_bias"] = position_bias

    baseline = _evaluate_baseline(
        scored_impressions=scored_impressions,
        teacher_item=teacher_item,
        news_meta=news_meta,
        k_out=k_out,
        position_bias=position_bias,
        category_target=fairness_base.get("category_target", "catalog"),
    )
    constraint = _make_constraint(baseline)
    seed = 13
    search_sample_size = 500
    if len(scored_impressions) > search_sample_size:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(
            rng.choice(len(scored_impressions), size=search_sample_size, replace=False)
        )
        scored_search = [scored_impressions[int(i)] for i in sample_idx.tolist()]
    else:
        scored_search = scored_impressions

    novelty_sims = ["teacher_cosine"]
    novelty_weights = [0.05, 0.10, 0.15]
    coverage_weights = [0.05, 0.10]
    fairness_penalties = [0.25, 0.50, 0.75]
    new_item_floors = [0.15, 0.20]

    sample_baseline = _evaluate_baseline(
        scored_impressions=scored_search,
        teacher_item=teacher_item,
        news_meta=news_meta,
        k_out=k_out,
        position_bias=position_bias,
        category_target=fairness_base.get("category_target", "catalog"),
    )

    sample_results = []
    search_space = []
    for novelty_sim in novelty_sims:
        for novelty_weight in novelty_weights:
            for coverage_weight in coverage_weights:
                relevance_weight = 1.0 - novelty_weight - coverage_weight
                if relevance_weight <= 0.0:
                    continue
                for penalty_weight in fairness_penalties:
                    for new_item_floor in new_item_floors:
                        search_space.append(
                            (
                                novelty_sim,
                                relevance_weight,
                                novelty_weight,
                                coverage_weight,
                                penalty_weight,
                                new_item_floor,
                            )
                        )

    for (
        novelty_sim,
        relevance_weight,
        novelty_weight,
        coverage_weight,
        penalty_weight,
        new_item_floor,
    ) in tqdm(search_space, desc="Search rerank grid"):
        fairness_cfg = dict(fairness_base)
        fairness_cfg["penalty_weight"] = penalty_weight
        fairness_cfg["new_item_floor"] = new_item_floor

        metrics = _evaluate_candidate(
            scored_impressions=scored_search,
            teacher_item=teacher_item,
            news_meta=news_meta,
            k_out=k_out,
            pool_size=pool_size,
            position_bias=position_bias,
            coverage_cfg=coverage_cfg,
            fairness_cfg=fairness_cfg,
            relevance_weight=relevance_weight,
            novelty_weight=novelty_weight,
            coverage_weight=coverage_weight,
            novelty_sim=novelty_sim,
        )
        metrics["constraint"] = _constraint_check(sample_baseline, metrics, constraint)
        sample_results.append(metrics)

    sample_results.sort(
        key=lambda r: (
            int(r["constraint"]["feasible"]),
            r["ndcg@k"],
            r["new_item_exposure_frac"],
            r["category_coverage"],
            -r["fairness_kl"],
        ),
        reverse=True,
    )

    shortlist = []
    seen = set()
    for item in sample_results:
        key = (
            item["novelty_sim"],
            item["weights"]["relevance"],
            item["weights"]["novelty"],
            item["weights"]["coverage"],
            item["fairness"]["penalty_weight"],
            item["fairness"]["new_item_floor"],
        )
        if key in seen:
            continue
        shortlist.append(item)
        seen.add(key)
        if len(shortlist) >= 1:
            break

    results = []
    for item in tqdm(shortlist, desc="Evaluate shortlist on full dev"):
        fairness_cfg = dict(fairness_base)
        fairness_cfg["penalty_weight"] = item["fairness"]["penalty_weight"]
        fairness_cfg["new_item_floor"] = item["fairness"]["new_item_floor"]
        metrics = _evaluate_candidate(
            scored_impressions=scored_impressions,
            teacher_item=teacher_item,
            news_meta=news_meta,
            k_out=k_out,
            pool_size=pool_size,
            position_bias=position_bias,
            coverage_cfg=coverage_cfg,
            fairness_cfg=fairness_cfg,
            relevance_weight=item["weights"]["relevance"],
            novelty_weight=item["weights"]["novelty"],
            coverage_weight=item["weights"]["coverage"],
            novelty_sim=item["novelty_sim"],
        )
        metrics["constraint"] = _constraint_check(baseline, metrics, constraint)
        results.append(metrics)

    feasible = [r for r in results if r["constraint"]["feasible"]]
    feasible.sort(
        key=lambda r: (
            r["ndcg@k"],
            r["new_item_exposure_frac"],
            r["category_coverage"],
            -r["fairness_kl"],
        ),
        reverse=True,
    )
    results.sort(
        key=lambda r: (
            int(r["constraint"]["feasible"]),
            r["ndcg@k"],
            r["new_item_exposure_frac"],
            r["category_coverage"],
            -r["fairness_kl"],
        ),
        reverse=True,
    )

    out = {
        "k_out": k_out,
        "pool_size": pool_size,
        "position_bias": position_bias,
        "baseline": baseline,
        "product_constraint": constraint,
        "search_sample_size": len(scored_search),
        "search_seed": seed,
        "n_candidates_screened": len(sample_results),
        "n_candidates_evaluated_full": len(results),
        "n_feasible": len(feasible),
        "best_feasible": feasible[0] if feasible else None,
        # top_10_sample: Best hyperparameter settings ranked on the sampled search subset used for cheap screening before full evaluation.
        # top_10:Best hyperparameter settings after reevaluating the shortlisted candidates on the full dev set.
        "top_10": results[:10],
        "top_10_sample": sample_results[:10],
    }
    save_json(out_root / "rerank_search.json", out)
