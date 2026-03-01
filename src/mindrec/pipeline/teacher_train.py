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
from mindrec.models.teacher import HistoryAttentionPool, l2_normalize
from mindrec.utils import set_seed, save_json


def run_train_teacher(cfg: dict[str, Any]) -> None:
    seed=int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

    ds=cfg["data"]["dataset_name"]
    proc_root=Path(cfg["data"]["processed_root"]) / ds
    runs_root=ensure_dir(Path("runs") / cfg["run_name"])
    art_root=ensure_dir(runs_root / "teacher")

    maps=IdMaps.load(proc_root / "id_maps.json")
    news=pd.read_parquet(proc_root / "news.parquet")

    model_name=cfg["teacher"]["model_name"]
    batch_size=int(cfg["teacher"]["batch_size"])
    device=cfg["teacher"].get("device", "cuda")
    if device=="cuda" and not torch.cuda.is_available():
        device="cpu"

    st=SentenceTransformer(model_name, device=device)

    texts=news["text"].fillna("").tolist()
    news_idx=news["news_idx"].astype(int).tolist()
    dim=st.get_sentence_embedding_dimension()

    item_emb=np.zeros((max(news_idx) + 1, dim), dtype=np.float32)  # row 0 reserved
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode news"):
        batch=texts[i:i + batch_size]
        emb=st.encode(batch, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        for j,e in enumerate(emb):
            item_emb[int(news_idx[i + j])]=e.astype(np.float32)

    np.save(art_root / "item_teacher_emb.npy", item_emb)

    # Build user embeddings from train histories (using raw train behaviors)
    raw_root=Path(cfg["data"]["raw_root"]) / cfg["data"]["train_dir"]
    beh=read_behaviors_tsv(raw_root / "behaviors.tsv")

    max_hist=int(cfg["data"]["max_history"])
    pooling=cfg["teacher"].get("user_pooling", "attention")

    user_emb=np.zeros((max(maps.user2idx.values()) + 1, dim), dtype=np.float32)
    if pooling=="mean":
        for _,r in tqdm(beh.iterrows(), total=len(beh), desc="User mean pooling"):
            u=str(r["user_id"])
            ui=maps.user2idx.get(u, 0)
            if ui==0:
                continue
            hist=[h for h in r["history"][-max_hist:] if h in maps.news2idx]
            if not hist:
                continue
            h_idx=[maps.news2idx.get(h, 0) for h in hist]
            vec=item_emb[h_idx].mean(axis=0)
            user_emb[ui]=l2_normalize(vec[None, :])[0]
    else:
        # Lightweight attention pooling (offline)
        attn=HistoryAttentionPool(dim=dim, heads=int(cfg["teacher"].get("user_attn_heads", 4))).to(device)
        attn.eval()
        for _,r in tqdm(beh.iterrows(), total=len(beh), desc="User attention pooling"):
            u=str(r["user_id"])
            ui=maps.user2idx.get(u, 0)
            if ui==0:
                continue
            hist=[h for h in r["history"][-max_hist:] if h in maps.news2idx]
            if not hist:
                continue
            h_idx=[maps.news2idx.get(h, 0) for h in hist]
            x=torch.tensor(item_emb[h_idx], dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,D]
            mask=torch.ones((1, x.size(1)), dtype=torch.bool, device=device)
            with torch.no_grad():
                pooled=attn(x, mask).cpu().numpy()
            user_emb[ui]=l2_normalize(pooled)[0]

    np.save(art_root / "user_teacher_emb.npy", user_emb)

    meta={
        "model_name": model_name,
        "dim": int(dim),
        "device": device,
        "pooling": pooling,
    }
    save_json(art_root / "meta.json", meta)
