from __future__ import annotations

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
from mindrec.metrics.ranking import (
    ndcg_at_k,
    recall_at_k,
    ndcg_from_order,
    recall_from_order,
)
from mindrec.pipeline.evaluate import _load_model
from mindrec.rerank.greedy import build_news_meta, cosine_sim_matrix, greedy_rerank
from mindrec.utils import position_bias_weights, save_json


def _cat_idx(news_meta: dict[str, Any], news_id: str) -> int:
    meta = news_meta.get(news_id)
    return int(meta.cat_idx) if meta is not None else 0


def _new_item_exposure_frac(
    weights: np.ndarray, ranking_idx: list[int], cand_is_new: list[int]
) -> float:
    new_exp = sum(
        float(weight)
        for weight, idx in zip(weights.tolist(), ranking_idx)
        if int(cand_is_new[idx]) == 1
    )
    return float(new_exp / (float(weights.sum()) + 1e-12))


def _category_target_dist(
    category_target: str, reference_cats: list[int]
) -> dict[int, float]:
    tgt = (
        uniform_target(reference_cats)
        if category_target == "uniform"
        else catalog_target(reference_cats)
    )
    return normalize_dist(tgt)


def run_rerank_eval(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    out_root = ensure_dir(runs_root / "eval")

    device_str = cfg["ranker"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model, teacher_user, teacher_item = _load_model(cfg, proc_root, runs_root, device)

    news = pd.read_parquet(proc_root / "news.parquet")
    news_meta = build_news_meta(news)

    impr = pd.read_parquet(proc_root / "dev_impressions.parquet")

    rr_cfg = cfg["rerank"]
    k_out = int(rr_cfg["k_out"])
    pool_size = int(rr_cfg["pool_size"])
    pos_mode = rr_cfg.get("position_bias", "log")

    rel_w = float(rr_cfg.get("relevance_weight", 0.85))
    nov_w = float(rr_cfg.get("novelty_weight", 0.10))
    cov_w = float(rr_cfg.get("coverage_weight", 0.05))
    novelty_sim = str(rr_cfg.get("novelty_sim", "teacher_cosine"))
    coverage_cfg = dict(rr_cfg.get("coverage", {}))
    fairness_cfg = dict(rr_cfg.get("fairness", {}))
    fairness_cfg["position_bias"] = pos_mode

    base_ndcg = []
    rr_ndcg = []
    base_recall = []
    rr_recall = []
    base_ild = []
    rr_ild = []
    base_cat_cov = []
    rr_cat_cov = []
    base_ent = []
    rr_ent = []
    base_fair_kl_full = []
    rr_fair_kl_full = []
    base_fair_kl_pool = []
    rr_fair_kl_pool = []
    base_fair_gini = []
    rr_fair_gini = []
    base_new_exp = []
    rr_new_exp = []

    with torch.no_grad():
        for _, r in tqdm(impr.iterrows(), total=len(impr), desc="Rerank eval"):
            labels = np.array(r["cand_label"], dtype=np.int32)
            if labels.sum() <= 0:
                continue
            user_idx = int(r["user_idx"])
            cand_news_id = list(r["cand_news_id"])
            cand_news_idx = np.array(r["cand_news_idx"], dtype=np.int64)
            cand_cat_idx = np.array(r["cand_cat_idx"], dtype=np.int64)
            cand_subcat_idx = np.array(r["cand_subcat_idx"], dtype=np.int64)
            cand_is_new = list(r["cand_is_new_item"])
            cand_clicks_log1p = np.array(r["cand_item_clicks_log1p"], dtype=np.float32)
            cand_cat_ref_full = [int(c) for c in cand_cat_idx.tolist() if int(c) != 0]
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

            # Score all candidates
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
                b_tu = tu[sl]
                b_ti = ti[sl]
                logit, _ = model(
                    user_idx=b_user,
                    news_idx=b_news,
                    cat_idx=b_cat,
                    subcat_idx=b_sub,
                    dense=b_dense,
                    teacher_user_emb=b_tu,
                    teacher_item_emb=b_ti,
                )
                logits.append(logit.detach().cpu().numpy())
            scores = np.concatenate(logits, axis=0)

            # Baseline: top-k_out by relevance
            base_order = np.argsort(-scores)[:k_out]
            base_ndcg.append(ndcg_at_k(labels, scores, k_out))
            base_recall.append(recall_at_k(labels, scores, k_out))

            base_ids = [cand_news_id[i] for i in base_order.tolist()]
            base_cats = [_cat_idx(news_meta, nid) for nid in base_ids]
            base_cat_cov.append(category_coverage([c for c in base_cats if c != 0]))
            base_ent.append(entropy([c for c in base_cats if c != 0]))

            # ILD via teacher cosine
            base_emb = teacher_item[cand_news_idx[base_order]]
            base_emb = base_emb / (
                np.linalg.norm(base_emb, axis=1, keepdims=True) + 1e-12
            )
            base_ild.append(ild_from_similarity(cosine_sim_matrix(base_emb)))

            # Exposure fairness (categories)
            w = position_bias_weights(k_out, mode=pos_mode)
            exp = normalize_dist(exposure_from_ranking(base_cats, w))
            pool_order = np.argsort(-scores)[:pool_size]
            cand_cat_ref_pool = [
                int(cand_cat_idx[i]) for i in pool_order.tolist() if int(cand_cat_idx[i]) != 0
            ]
            tgt_full = _category_target_dist(
                fairness_cfg.get("category_target", "catalog"), cand_cat_ref_full
            )
            tgt_pool = _category_target_dist(
                fairness_cfg.get("category_target", "catalog"), cand_cat_ref_pool
            )
            base_fair_kl_full.append(kl_divergence(exp, tgt_full))
            base_fair_kl_pool.append(kl_divergence(exp, tgt_pool))
            base_fair_gini.append(gini(list(exp.values())))

            base_new_exp.append(
                _new_item_exposure_frac(w, base_order.tolist(), cand_is_new)
            )

            # Re-rank
            pool_emb = teacher_item[cand_news_idx[pool_order]]
            rr = greedy_rerank(
                cand_news_id=cand_news_id,
                cand_scores=scores,
                cand_is_new=cand_is_new,
                news_meta=news_meta,
                item_teacher_emb=pool_emb,
                k_out=k_out,
                pool_size=pool_size,
                relevance_weight=rel_w,
                novelty_weight=nov_w,
                coverage_weight=cov_w,
                novelty_sim=novelty_sim,
                coverage_cfg=coverage_cfg,
                fairness_cfg=fairness_cfg,
            )
            rr_ids = rr["ranked_news_id"]
            cand_idx_by_id = {nid: idx for idx, nid in enumerate(cand_news_id)}
            rr_idx = [cand_idx_by_id[nid] for nid in rr_ids]
            rr_ndcg.append(ndcg_from_order(labels, np.array(rr_idx), k_out))
            rr_recall.append(recall_from_order(labels, np.array(rr_idx), k_out))

            rr_cats = [_cat_idx(news_meta, nid) for nid in rr_ids]
            rr_cat_cov.append(category_coverage([c for c in rr_cats if c != 0]))
            rr_ent.append(entropy([c for c in rr_cats if c != 0]))

            rr_emb = teacher_item[cand_news_idx[rr_idx]]
            rr_emb = rr_emb / (np.linalg.norm(rr_emb, axis=1, keepdims=True) + 1e-12)
            rr_ild.append(ild_from_similarity(cosine_sim_matrix(rr_emb)))

            exp2 = normalize_dist(exposure_from_ranking(rr_cats, w))
            rr_fair_kl_full.append(kl_divergence(exp2, tgt_full))
            rr_fair_kl_pool.append(kl_divergence(exp2, tgt_pool))
            rr_fair_gini.append(gini(list(exp2.values())))

            rr_new_exp.append(_new_item_exposure_frac(w, rr_idx, cand_is_new))

    out = {
        "k_out": k_out,
        "pool_size": pool_size,
        "baseline": {
            "ndcg@k": float(np.mean(base_ndcg) if base_ndcg else 0.0),
            "recall@k": float(np.mean(base_recall) if base_recall else 0.0),
            "ild": float(np.mean(base_ild) if base_ild else 0.0),
            "category_coverage": float(np.mean(base_cat_cov) if base_cat_cov else 0.0),
            "category_entropy": float(np.mean(base_ent) if base_ent else 0.0),
            "fairness_kl_full": float(
                np.mean(base_fair_kl_full) if base_fair_kl_full else 0.0
            ),
            "fairness_kl_pool": float(
                np.mean(base_fair_kl_pool) if base_fair_kl_pool else 0.0
            ),
            "fairness_gini": float(np.mean(base_fair_gini) if base_fair_gini else 0.0),
            "new_item_exposure_frac": float(
                np.mean(base_new_exp) if base_new_exp else 0.0
            ),
        },
        "reranked": {
            "ndcg@k": float(np.mean(rr_ndcg) if rr_ndcg else 0.0),
            "recall@k": float(np.mean(rr_recall) if rr_recall else 0.0),
            "ild": float(np.mean(rr_ild) if rr_ild else 0.0),
            "category_coverage": float(np.mean(rr_cat_cov) if rr_cat_cov else 0.0),
            "category_entropy": float(np.mean(rr_ent) if rr_ent else 0.0),
            "fairness_kl_full": float(
                np.mean(rr_fair_kl_full) if rr_fair_kl_full else 0.0
            ),
            "fairness_kl_pool": float(
                np.mean(rr_fair_kl_pool) if rr_fair_kl_pool else 0.0
            ),
            "fairness_gini": float(np.mean(rr_fair_gini) if rr_fair_gini else 0.0),
            "new_item_exposure_frac": float(np.mean(rr_new_exp) if rr_new_exp else 0.0),
        },
        "weights": {
            "relevance": rel_w,
            "novelty": nov_w,
            "coverage": cov_w,
        },
        "novelty_sim": novelty_sim,
        "position_bias": pos_mode,
    }
    save_json(out_root / "rerank_eval.json", out)
