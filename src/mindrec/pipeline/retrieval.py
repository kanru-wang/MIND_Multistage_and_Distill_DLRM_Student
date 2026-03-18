from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.data.mind_io import read_behaviors_tsv
from mindrec.models.teacher import TeacherTwoTower
from mindrec.utils import save_json


def _build_index(item_emb: np.ndarray, index_type: str, ivf_nlist: int) -> faiss.Index:
    dim = item_emb.shape[1]
    xb = item_emb.astype(np.float32)
    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(dim)
        index.add(xb)
        return index

    if index_type == "ivf_flat_ip":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer, dim, ivf_nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.train(xb)
        index.add(xb)
        return index

    raise ValueError(f"Unknown index_type: {index_type}")


def run_build_index(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "retrieval")

    teacher_root = runs_root / "teacher"
    item_emb = np.load(teacher_root / "item_teacher_emb.npy")

    index = _build_index(
        item_emb=item_emb,
        index_type=cfg["retrieval"]["index_type"],
        ivf_nlist=int(cfg["retrieval"].get("ivf_nlist", 2048)),
    )
    faiss.write_index(index, str(art_root / "faiss.index"))
    save_json(
        art_root / "meta.json",
        {
            "index_type": cfg["retrieval"]["index_type"],
            "ivf_nlist": int(cfg["retrieval"].get("ivf_nlist", 2048)),
            "n_items": int(item_emb.shape[0]),
            "dim": int(item_emb.shape[1]),
        },
    )


def run_eval_retrieval(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "retrieval")

    teacher_root = runs_root / "teacher"
    item_emb = np.load(teacher_root / "item_teacher_emb.npy")
    item_emb_tensor = torch.tensor(item_emb, dtype=torch.float32)
    model_ckpt_path = teacher_root / "model.pt"

    index = faiss.read_index(str(art_root / "faiss.index"))
    topk = int(cfg["retrieval"]["topk"])
    max_hist = int(cfg["data"]["max_history"])

    recalls = []
    if model_ckpt_path.exists():
        ckpt = torch.load(model_ckpt_path, map_location="cpu")
        model = TeacherTwoTower(
            item_dim=int(ckpt["item_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            heads=int(ckpt["heads"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        raw_root = Path(cfg["data"]["raw_root"]) / cfg["data"]["dev_dir"]
        beh_dev = read_behaviors_tsv(raw_root / "behaviors.tsv")
        maps = IdMaps.load(proc_root / "id_maps.json")

        for _, r in tqdm(beh_dev.iterrows(), total=len(beh_dev), desc="Retrieval eval"):
            hist_idx = [
                maps.news2idx[h]
                for h in r["history"][-max_hist:]
                if h in maps.news2idx and maps.news2idx[h] != 0
            ]
            if not hist_idx:
                continue

            clicked = [
                maps.news2idx.get(str(n), 0)
                for n, l in zip(r["cand_news_id"], r["cand_label"])
                if int(l) == 1 and maps.news2idx.get(str(n), 0) != 0
            ]
            if not clicked:
                continue

            with torch.no_grad():
                hist_z = item_emb_tensor[hist_idx].unsqueeze(0)
                hist_mask = torch.ones((1, len(hist_idx)), dtype=torch.bool)
                q = (
                    model.encode_user_from_item_vectors(hist_z, hist_mask)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            _, I = index.search(q, topk)
            retrieved = set(I[0].tolist())
            hit = sum(1 for c in clicked if c in retrieved)
            recalls.append(hit / len(clicked))
    else:
        impr = pd.read_parquet(proc_root / "dev_impressions.parquet")
        user_emb = np.load(teacher_root / "user_teacher_emb.npy")

        for _, r in tqdm(impr.iterrows(), total=len(impr), desc="Retrieval eval"):
            u = int(r["user_idx"])
            if u <= 0:
                continue
            q = user_emb[u : u + 1].astype(np.float32)
            _, I = index.search(q, topk)
            retrieved = set(I[0].tolist())
            clicked = [
                int(n) for n, l in zip(r["cand_news_idx"], r["cand_label"]) if int(l) == 1
            ]
            if not clicked:
                continue
            hit = sum(1 for c in clicked if c in retrieved)
            recalls.append(hit / len(clicked))

    out = {
        "recall_at_k": float(np.mean(recalls) if recalls else 0.0),
        "n_eval": int(len(recalls)),
        "k": topk,
    }
    save_json(art_root / "eval.json", out)
