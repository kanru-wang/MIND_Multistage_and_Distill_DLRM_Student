from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.metrics.calibration import brier_score, expected_calibration_error
from mindrec.metrics.ranking import auc, average_precision_at_k, mrr, ndcg_at_k, recall_at_k
from mindrec.models.dlrm import DLRMStudent
from mindrec.utils import position_bias_weights, save_json


def _load_model(cfg: dict[str, Any], proc_root: Path, runs_root: Path, device: torch.device) -> tuple[DLRMStudent, np.ndarray, np.ndarray]:
    maps=IdMaps.load(proc_root / "id_maps.json")
    news=pd.read_parquet(proc_root / "news.parquet")
    n_users=max(maps.user2idx.values()) + 1
    n_news=int(news["news_idx"].max()) + 1
    n_cats=int(news["cat_idx"].max()) + 1
    n_subcats=int(news["subcat_idx"].max()) + 1

    teacher_item=np.load(runs_root / "teacher" / "item_teacher_emb.npy")
    teacher_dim=int(teacher_item.shape[1])
    teacher_user=np.load(runs_root / "teacher" / "user_teacher_emb.npy")

    dlrm_cfg=cfg["ranker"]["dlrm"]
    model=DLRMStudent(
        n_users=n_users,
        n_news=n_news,
        n_cats=n_cats,
        n_subcats=n_subcats,
        dense_dim=2,
        emb_dim=int(dlrm_cfg["emb_dim"]),
        bottom_mlp=[int(x) for x in dlrm_cfg["bottom_mlp"]],
        top_mlp=[int(x) for x in dlrm_cfg["top_mlp"]],
        dropout=float(dlrm_cfg.get("dropout", 0.0)),
        teacher_dim=teacher_dim,
        fusion_heads=4,
    ).to(device)

    ckpt=torch.load(runs_root / "ranker" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, teacher_user, teacher_item


def run_evaluate(cfg: dict[str, Any]) -> None:
    ds=cfg["data"]["dataset_name"]
    proc_root=Path(cfg["data"]["processed_root"]) / ds
    runs_root=ensure_dir(Path("runs") / cfg["run_name"])
    out_root=ensure_dir(runs_root / "eval")

    device_str=cfg["ranker"].get("device", "cuda")
    if device_str=="cuda" and not torch.cuda.is_available():
        device_str="cpu"
    device=torch.device(device_str)

    model, teacher_user, teacher_item=_load_model(cfg, proc_root, runs_root, device)

    impr=pd.read_parquet(proc_root / "dev_impressions.parquet")
    ks=[int(k) for k in cfg["eval"]["ks"]]

    # Aggregate metrics overall and slices
    agg={f"ndcg@{k}":[] for k in ks}
    agg.update({f"recall@{k}":[] for k in ks})
    agg.update({f"map@{k}":[] for k in ks})
    agg["mrr"]=[]
    agg["auc"]=[]
    all_probs=[]
    all_labels=[]

    slice_defs={
        "overall": lambda r: True,
        "cold_user": lambda r: int(r["is_cold_user"])==1,
        "warm_user": lambda r: int(r["is_cold_user"])==0,
    }
    slice_aggs={name:{k:[] for k in agg.keys()} for name in slice_defs.keys()}

    with torch.no_grad():
        for _,r in tqdm(impr.iterrows(), total=len(impr), desc="Eval ranker"):
            labels=np.array(r["cand_label"], dtype=np.int32)
            if labels.sum() <= 0:
                continue
            user_idx=int(r["user_idx"])
            cand_news_idx=np.array(r["cand_news_idx"], dtype=np.int64)
            cand_cat_idx=np.array(r["cand_cat_idx"], dtype=np.int64)
            cand_subcat_idx=np.array(r["cand_subcat_idx"], dtype=np.int64)
            cand_clicks_log1p=np.array(r["cand_item_clicks_log1p"], dtype=np.float32)

            # Dense: [history_len, item_clicks_log1p]
            hlen=float(r["history_len"])
            dense=np.stack([np.full_like(cand_clicks_log1p, hlen), cand_clicks_log1p], axis=1)

            # Teacher embeddings
            tu=torch.tensor(teacher_user[user_idx:user_idx + 1], dtype=torch.float32, device=device).repeat(len(cand_news_idx), 1)
            ti=torch.tensor(teacher_item[cand_news_idx], dtype=torch.float32, device=device)

            logits=[]
            bs=2048
            for i in range(0, len(cand_news_idx), bs):
                sl=slice(i, i + bs)
                b_user=torch.tensor([user_idx] * len(cand_news_idx[sl]), dtype=torch.long, device=device)
                b_news=torch.tensor(cand_news_idx[sl], dtype=torch.long, device=device)
                b_cat=torch.tensor(cand_cat_idx[sl], dtype=torch.long, device=device)
                b_sub=torch.tensor(cand_subcat_idx[sl], dtype=torch.long, device=device)
                b_dense=torch.tensor(dense[sl], dtype=torch.float32, device=device)
                b_tu=tu[sl]
                b_ti=ti[sl]
                logit,_=model(
                    user_idx=b_user,
                    news_idx=b_news,
                    cat_idx=b_cat,
                    subcat_idx=b_sub,
                    dense=b_dense,
                    teacher_user_emb=b_tu,
                    teacher_item_emb=b_ti,
                )
                logits.append(logit.detach().cpu().numpy())
            scores=np.concatenate(logits, axis=0)
            probs=1.0 / (1.0 + np.exp(-scores))

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.astype(float).tolist())

            # Per-impression metrics
            m=mrr(labels, scores)
            a=auc(labels, scores)
            agg["mrr"].append(m)
            agg["auc"].append(a)
            for k in ks:
                agg[f"ndcg@{k}"].append(ndcg_at_k(labels, scores, k))
                agg[f"recall@{k}"].append(recall_at_k(labels, scores, k))
                agg[f"map@{k}"].append(average_precision_at_k(labels, scores, k))

            # Slice metrics
            for name,fn in slice_defs.items():
                if fn(r):
                    slice_aggs[name]["mrr"].append(m)
                    slice_aggs[name]["auc"].append(a)
                    for k in ks:
                        slice_aggs[name][f"ndcg@{k}"].append(ndcg_at_k(labels, scores, k))
                        slice_aggs[name][f"recall@{k}"].append(recall_at_k(labels, scores, k))
                        slice_aggs[name][f"map@{k}"].append(average_precision_at_k(labels, scores, k))

    y=np.array(all_labels, dtype=np.float32)
    p=np.array(all_probs, dtype=np.float32)

    out={
        "ranking": {k: float(np.mean(v) if v else 0.0) for k,v in agg.items()},
        "calibration": {
            "brier": brier_score(y, p),
            "ece_15": expected_calibration_error(y, p, n_bins=15),
        },
        "slices": {
            name: {k: float(np.mean(v) if v else 0.0) for k,v in d.items()}
            for name,d in slice_aggs.items()
        },
        "n_impressions": int(len(impr)),
        "n_scored_pairs": int(len(y)),
        "device": device_str,
    }
    save_json(out_root / "ranker_eval.json", out)
