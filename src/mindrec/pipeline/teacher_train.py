from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.data.mind_io import read_behaviors_tsv
from mindrec.models.teacher import TeacherTwoTower
from mindrec.utils import save_json, set_seed


@dataclass
class TeacherSample:
    user_idx: int
    pos_news_idx: int
    hist_news_idx: list[int]


class TeacherDataset(Dataset):
    def __init__(self, samples: list[TeacherSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TeacherSample:
        return self.samples[idx]


def _collate_teacher_batch(batch: list[TeacherSample]) -> dict[str, torch.Tensor]:
    max_len = max(len(sample.hist_news_idx) for sample in batch)
    hist = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, sample in enumerate(batch):
        n = len(sample.hist_news_idx)
        hist[i, :n] = torch.tensor(sample.hist_news_idx, dtype=torch.long)
        mask[i, :n] = True
    return {
        "user_idx": torch.tensor([sample.user_idx for sample in batch], dtype=torch.long),
        "pos_news_idx": torch.tensor(
            [sample.pos_news_idx for sample in batch], dtype=torch.long
        ),
        "hist_news_idx": hist,
        "hist_mask": mask,
    }


def _build_teacher_samples(
    beh: pd.DataFrame, maps: IdMaps, max_hist: int
) -> tuple[list[TeacherSample], dict[int, list[int]]]:
    samples: list[TeacherSample] = []
    best_hist_by_user: dict[int, list[int]] = {}

    for _, row in tqdm(beh.iterrows(), total=len(beh), desc="Build teacher samples"):
        user_idx = maps.user2idx.get(str(row["user_id"]), 0)
        if user_idx == 0:
            continue

        hist_news_idx = [
            maps.news2idx[h]
            for h in row["history"][-max_hist:]
            if h in maps.news2idx and maps.news2idx[h] != 0
        ]
        if not hist_news_idx:
            continue

        prev = best_hist_by_user.get(user_idx)
        if prev is None or len(hist_news_idx) > len(prev):
            best_hist_by_user[user_idx] = hist_news_idx

        for news_id, label in zip(row["cand_news_id"], row["cand_label"]):
            if int(label) != 1:
                continue
            news_idx = maps.news2idx.get(str(news_id), 0)
            if news_idx == 0:
                continue
            samples.append(
                TeacherSample(
                    user_idx=user_idx,
                    pos_news_idx=news_idx,
                    hist_news_idx=hist_news_idx,
                )
            )

    return samples, best_hist_by_user


def _encode_news_text(
    st: SentenceTransformer,
    news: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    texts = news["text"].fillna("").tolist()
    news_idx = news["news_idx"].astype(int).tolist()
    dim = int(st.get_sentence_embedding_dimension())
    item_emb = np.zeros((max(news_idx) + 1, dim), dtype=np.float32)

    for i in tqdm(range(0, len(texts), batch_size), desc="Encode news"):
        batch = texts[i : i + batch_size]
        emb = st.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for j, vec in enumerate(emb):
            item_emb[int(news_idx[i + j])] = vec.astype(np.float32)
    return item_emb


def _compute_user_embeddings(
    model: TeacherTwoTower,
    item_base: np.ndarray,
    histories_by_user: dict[int, list[int]],
    n_users: int,
    device: torch.device,
) -> np.ndarray:
    hidden_dim = model.item_proj[-1].out_features
    user_emb = np.zeros((n_users, hidden_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for user_idx, hist_idx in tqdm(
            histories_by_user.items(),
            total=len(histories_by_user),
            desc="Encode users",
        ):
            hist_base = torch.tensor(
                item_base[hist_idx], dtype=torch.float32, device=device
            ).unsqueeze(0)
            hist_mask = torch.ones((1, len(hist_idx)), dtype=torch.bool, device=device)
            user_vec = model.encode_user(hist_base, hist_mask)
            user_emb[user_idx] = user_vec.squeeze(0).detach().cpu().numpy()
    return user_emb


def run_train_teacher(cfg: dict[str, Any]) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "teacher")

    maps = IdMaps.load(proc_root / "id_maps.json")
    news = pd.read_parquet(proc_root / "news.parquet")

    teacher_cfg = dict(cfg["teacher"])
    batch_size = int(teacher_cfg["batch_size"])
    epochs = int(teacher_cfg.get("epochs", 1))
    lr = float(teacher_cfg.get("lr", 2.0e-4))
    hidden_dim = int(teacher_cfg.get("user_attn_dim", 256))
    heads = int(teacher_cfg.get("user_attn_heads", 4))
    device_str = str(teacher_cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    st = SentenceTransformer(teacher_cfg["model_name"], device=device_str)
    item_base = _encode_news_text(st=st, news=news, batch_size=batch_size)

    raw_root = Path(cfg["data"]["raw_root"]) / cfg["data"]["train_dir"]
    beh_train = read_behaviors_tsv(raw_root / "behaviors.tsv")
    max_hist = int(cfg["data"]["max_history"])
    samples, histories_by_user = _build_teacher_samples(beh_train, maps, max_hist=max_hist)
    if not samples:
        raise ValueError("No teacher training samples were built from train behaviors.")

    train_ds = TeacherDataset(samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_teacher_batch,
    )

    item_base_tensor = torch.tensor(item_base, dtype=torch.float32, device=device)
    model = TeacherTwoTower(
        item_dim=item_base.shape[1],
        hidden_dim=hidden_dim,
        heads=heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-6)

    train_loss_mean = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Train teacher ep {epoch}"):
            pos_idx = batch["pos_news_idx"].to(device)
            hist_idx = batch["hist_news_idx"].to(device)
            hist_mask = batch["hist_mask"].to(device)

            pos_base = item_base_tensor[pos_idx]
            hist_base = item_base_tensor[hist_idx]

            user_z = model.encode_user(hist_base, hist_mask)
            item_z = model.encode_items(pos_base)
            logits = user_z @ item_z.T
            targets = torch.arange(logits.size(0), device=device)
            loss_u = F.cross_entropy(logits, targets)
            loss_i = F.cross_entropy(logits.T, targets)
            loss = 0.5 * (loss_u + loss_i)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.item()))

        train_loss_mean = float(np.mean(losses) if losses else 0.0)

    model.eval()
    with torch.no_grad():
        item_teacher_emb = model.encode_items(item_base_tensor).detach().cpu().numpy()
    user_teacher_emb = _compute_user_embeddings(
        model=model,
        item_base=item_base,
        histories_by_user=histories_by_user,
        n_users=max(maps.user2idx.values(), default=0) + 1,
        device=device,
    )

    np.save(art_root / "item_teacher_emb.npy", item_teacher_emb.astype(np.float32))
    np.save(art_root / "user_teacher_emb.npy", user_teacher_emb.astype(np.float32))

    meta = {
        "model_name": teacher_cfg["model_name"],
        "device": device_str,
        "item_base_dim": int(item_base.shape[1]),
        "teacher_dim": int(item_teacher_emb.shape[1]),
        "hidden_dim": hidden_dim,
        "user_pooling": "attention",
        "train_samples": int(len(samples)),
        "train_users": int(len(histories_by_user)),
        "epochs": epochs,
        "lr": lr,
        "train_loss_mean": train_loss_mean,
        "item_encoder": "sentence_transformer_frozen_plus_projection",
        "user_encoder": "attention_pool_over_history",
    }
    save_json(art_root / "meta.json", meta)
