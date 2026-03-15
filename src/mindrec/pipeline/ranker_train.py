from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.datasets import PairDataset, collate_batch
from mindrec.data.featurize import IdMaps
from mindrec.models.calibration import fit_temperature_scaler
from mindrec.models.dlrm import DLRMStudent
from mindrec.models.distill import pairwise_logit_distill_bce, repr_distill_mse
from mindrec.utils import set_seed, save_json, to_device


class StudentProjHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def run_train_ranker(cfg: dict[str, Any]) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    maps = IdMaps.load(proc_root / "id_maps.json")

    pairs_train = pd.read_parquet(proc_root / "train_pairs.parquet")
    pairs_dev = pd.read_parquet(proc_root / "dev_pairs.parquet")

    dense_cols = ["history_len", "item_clicks_log1p"]
    train_ds = PairDataset(pairs_train, dense_cols=dense_cols)
    dev_ds = PairDataset(pairs_dev, dense_cols=dense_cols)

    device_str = cfg["ranker"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "ranker")

    teacher_user = np.load(runs_root / "teacher" / "user_teacher_emb.npy")
    teacher_item = np.load(runs_root / "teacher" / "item_teacher_emb.npy")
    teacher_dim = int(teacher_item.shape[1])

    news = pd.read_parquet(proc_root / "news.parquet")
    n_users = max(maps.user2idx.values()) + 1
    n_news = int(news["news_idx"].max()) + 1
    n_cats = int(news["cat_idx"].max()) + 1
    n_subcats = int(news["subcat_idx"].max()) + 1

    dlrm_cfg = cfg["ranker"]["dlrm"]
    model = DLRMStudent(
        n_users=n_users,
        n_news=n_news,
        n_cats=n_cats,
        n_subcats=n_subcats,
        dense_dim=len(dense_cols),
        emb_dim=int(dlrm_cfg["emb_dim"]),
        bottom_mlp=[int(x) for x in dlrm_cfg["bottom_mlp"]],
        top_mlp=[int(x) for x in dlrm_cfg["top_mlp"]],
        dropout=float(dlrm_cfg.get("dropout", 0.0)),
        teacher_dim=teacher_dim,
        fusion_heads=4,
    ).to(device)

    # Student repr dimension: eu(emb_dim)+ei(emb_dim)+zf(emb_dim)=3*emb_dim
    emb_dim = int(dlrm_cfg["emb_dim"])
    student_repr_dim = 3 * emb_dim
    teacher_repr_dim = 2 * teacher_dim
    proj = StudentProjHead(student_repr_dim, teacher_repr_dim).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(proj.parameters()),
        lr=float(cfg["ranker"]["lr"]),
        weight_decay=float(cfg["ranker"].get("weight_decay", 0.0)),
    )

    bsz = int(cfg["ranker"]["batch_size"])
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True, num_workers=0, collate_fn=collate_batch
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=bsz, shuffle=False, num_workers=0, collate_fn=collate_batch
    )

    dist_cfg = cfg["ranker"]["distill"]
    dist_enabled = bool(dist_cfg.get("enabled", True))
    temp = float(dist_cfg.get("temperature", 2.0))
    lam_logit = float(dist_cfg.get("lambda_logit", 1.0))
    lam_repr = float(dist_cfg.get("lambda_repr", 0.1))
    w_cold = float(dist_cfg.get("cold_weight", 2.0))
    w_warm = float(dist_cfg.get("warm_weight", 0.3))

    epochs = int(cfg["ranker"]["epochs"])

    best_auc = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        proj.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Train ep {ep}"):
            batch = to_device(batch, device)
            ui = batch["user_idx"].cpu().numpy()
            ni = batch["news_idx"].cpu().numpy()
            tu = torch.tensor(teacher_user[ui], dtype=torch.float32, device=device)
            ti = torch.tensor(teacher_item[ni], dtype=torch.float32, device=device)

            logits, rep = model(
                user_idx=batch["user_idx"],
                news_idx=batch["news_idx"],
                cat_idx=batch["cat_idx"],
                subcat_idx=batch["subcat_idx"],
                dense=batch["dense"],
                teacher_user_emb=tu,
                teacher_item_emb=ti,
                return_repr=True,
            )
            y = batch["label"]
            loss_rank = nn.functional.binary_cross_entropy_with_logits(logits, y)

            loss = loss_rank
            if dist_enabled:
                # teacher logit as cosine / inner product (embeddings are normalized)
                tlogit = (tu * ti).sum(dim=1)
                loss_logit = pairwise_logit_distill_bce(
                    logits, tlogit, temperature=temp
                )

                t_repr = torch.cat([tu, ti], dim=1)
                s_repr = proj(rep)
                loss_repr = repr_distill_mse(s_repr, t_repr)

                cold_mask = (
                    (batch["is_cold_user"] == 1) | (batch["is_new_item"] == 1)
                ).float()
                w = (cold_mask * w_cold) + ((1.0 - cold_mask) * w_warm)
                w = w.detach()

                loss = loss + (
                    w.mean() * (lam_logit * loss_logit + lam_repr * loss_repr)
                )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(proj.parameters()), max_norm=5.0
            )
            opt.step()
            losses.append(float(loss.item()))

        # Dev AUC (pairwise)
        model.eval()
        proj.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Dev ep {ep}"):
                batch = to_device(batch, device)
                ui = batch["user_idx"].cpu().numpy()
                ni = batch["news_idx"].cpu().numpy()
                tu = torch.tensor(teacher_user[ui], dtype=torch.float32, device=device)
                ti = torch.tensor(teacher_item[ni], dtype=torch.float32, device=device)
                logits, _ = model(
                    user_idx=batch["user_idx"],
                    news_idx=batch["news_idx"],
                    cat_idx=batch["cat_idx"],
                    subcat_idx=batch["subcat_idx"],
                    dense=batch["dense"],
                    teacher_user_emb=tu,
                    teacher_item_emb=ti,
                    return_repr=False,
                )
                ys.extend(batch["label"].detach().cpu().numpy().tolist())
                ps.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        try:
            auc = float(roc_auc_score(ys, ps))
        except ValueError:
            auc = 0.0

        save_json(
            art_root / f"epoch_{ep}.json",
            {
                "epoch": ep,
                "train_loss_mean": float(np.mean(losses) if losses else 0.0),
                "dev_auc": auc,
            },
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "proj": proj.state_dict(),
                    "cfg": cfg,
                },
                art_root / "best.pt",
            )

    save_json(art_root / "train_summary.json", {"best_dev_auc": best_auc})

    cal_cfg = dict(cfg["ranker"].get("calibration", {}))
    if bool(cal_cfg.get("enabled", True)):
        ckpt = torch.load(art_root / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Fit temperature"):
                batch = to_device(batch, device)
                ui = batch["user_idx"].cpu().numpy()
                ni = batch["news_idx"].cpu().numpy()
                tu = torch.tensor(teacher_user[ui], dtype=torch.float32, device=device)
                ti = torch.tensor(teacher_item[ni], dtype=torch.float32, device=device)
                logits, _ = model(
                    user_idx=batch["user_idx"],
                    news_idx=batch["news_idx"],
                    cat_idx=batch["cat_idx"],
                    subcat_idx=batch["subcat_idx"],
                    dense=batch["dense"],
                    teacher_user_emb=tu,
                    teacher_item_emb=ti,
                    return_repr=False,
                )
                logits_all.append(logits.detach().cpu().numpy())
                labels_all.append(batch["label"].detach().cpu().numpy())

        scaler, stats = fit_temperature_scaler(
            logits=np.concatenate(logits_all, axis=0),
            labels=np.concatenate(labels_all, axis=0),
            max_iter=int(cal_cfg.get("max_iter", 100)),
            lr=float(cal_cfg.get("lr", 0.05)),
        )
        scaler.save(
            art_root / "calibration.json",
            meta={
                "fit_split": "dev_pairs",
                "stats": stats,
            },
        )
        save_json(art_root / "calibration_stats.json", stats)
