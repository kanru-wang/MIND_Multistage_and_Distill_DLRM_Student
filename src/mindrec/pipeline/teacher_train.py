from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.data.mind_io import read_behaviors_tsv
from mindrec.models.teacher import l2_normalize
from mindrec.utils import set_seed, save_json


def _smart_history_pool(item_emb: np.ndarray, h_idx: list[int]) -> np.ndarray:
    """Train-free weighted pooling: recency + consistency with history centroid."""
    x = item_emb[h_idx]  # [T, D], sentence-transformer vectors are already normalized.
    t = x.shape[0]
    if t == 1:
        return x[0]

    # Newer clicks get higher weight.
    recency = np.linspace(0.2, 1.0, t, dtype=np.float32)

    # Items aligned with the user's history centroid get higher weight.
    centroid = l2_normalize(x.mean(axis=0, keepdims=True))[0]  # [D]
    consistency = np.clip(((x * centroid[None, :]).sum(axis=1) + 1.0) * 0.5, 0.0, 1.0)

    w = recency * (0.5 + 0.5 * consistency)
    w = w / (w.sum() + 1e-12)
    vec = (x * w[:, None]).sum(axis=0)
    return l2_normalize(vec[None, :])[0]


def run_train_teacher(cfg: dict[str, Any]) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "teacher")

    maps = IdMaps.load(proc_root / "id_maps.json")
    news = pd.read_parquet(proc_root / "news.parquet")

    model_name = cfg["teacher"]["model_name"]
    batch_size = int(cfg["teacher"]["batch_size"])
    device = cfg["teacher"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    st = SentenceTransformer(model_name, device=device)

    texts = news["text"].fillna("").tolist()
    news_idx = news["news_idx"].astype(int).tolist()
    dim = st.get_sentence_embedding_dimension()

    item_emb = np.zeros((max(news_idx) + 1, dim), dtype=np.float32)  # row 0 reserved
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode news"):
        batch = texts[i : i + batch_size]
        emb = st.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for j, e in enumerate(emb):
            item_emb[int(news_idx[i + j])] = e.astype(np.float32)

    np.save(art_root / "item_teacher_emb.npy", item_emb)

    # Build user embeddings from train histories (using raw train behaviors)
    raw_root = Path(cfg["data"]["raw_root"]) / cfg["data"]["train_dir"]
    beh = read_behaviors_tsv(raw_root / "behaviors.tsv")

    max_hist = int(cfg["data"]["max_history"])
    pooling = str(cfg["teacher"].get("user_pooling", "smart")).lower().strip()
    if pooling == "attention":
        # Legacy alias: prior "attention" mode used untrained random attention weights.
        # Map it to a deterministic, stronger train-free pooling strategy.
        pooling = "smart"
    if pooling not in {"mean", "smart"}:
        raise ValueError(f"Unknown teacher.user_pooling mode: {pooling}")

    user_emb = np.zeros((max(maps.user2idx.values()) + 1, dim), dtype=np.float32)
    # Aggregate one robust history per user to avoid repeatedly overwriting user vectors.
    user_hist_idx: dict[int, list[int]] = {}
    for _, r in tqdm(beh.iterrows(), total=len(beh), desc="Collect user histories"):
        u = str(r["user_id"])
        ui = maps.user2idx.get(u, 0)
        if ui == 0:
            continue
        hist = [h for h in r["history"][-max_hist:] if h in maps.news2idx]
        if not hist:
            continue
        h_idx = [maps.news2idx.get(h, 0) for h in hist]
        prev = user_hist_idx.get(ui)
        if prev is None or len(h_idx) > len(prev):
            user_hist_idx[ui] = h_idx

    desc = "User mean pooling" if pooling == "mean" else "User smart pooling"
    for ui, h_idx in tqdm(user_hist_idx.items(), total=len(user_hist_idx), desc=desc):
        if pooling == "mean":
            vec = item_emb[h_idx].mean(axis=0)
            user_emb[ui] = l2_normalize(vec[None, :])[0]
        else:
            user_emb[ui] = _smart_history_pool(item_emb, h_idx)

    np.save(art_root / "user_teacher_emb.npy", user_emb)

    meta = {
        "model_name": model_name,
        "dim": int(dim),
        "device": device,
        "pooling": pooling,
    }
    save_json(art_root / "meta.json", meta)
