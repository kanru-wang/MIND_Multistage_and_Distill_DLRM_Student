from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps, add_indices, build_id_maps, is_cold_user
from mindrec.data.mind_io import read_behaviors_tsv, read_news_tsv, sub_sample_behaviors
from mindrec.utils import set_seed, save_json


def build_pairs(
    beh: pd.DataFrame,
    news_idx_df: pd.DataFrame,
    maps: IdMaps,
    item_clicks_train: dict[str, int],
    min_user_hist_for_warm: int,
    min_item_train_clicks_for_warm: int,
    neg_per_pos: int = 4,
    seed: int = 13,
) -> pd.DataFrame:
    set_seed(seed)
    news_lookup = news_idx_df.set_index("news_id")[
        ["news_idx", "cat_idx", "subcat_idx"]
    ].to_dict(orient="index")

    rows = []
    for _, r in tqdm(beh.iterrows(), total=len(beh), desc="Build pairs"):
        user_id = str(r["user_id"])
        user_idx = maps.user2idx.get(user_id, 0)
        hist = r["history"]
        cold_u = 1 if is_cold_user(hist, min_user_hist_for_warm) else 0

        cand_ids = list(r["cand_news_id"])
        labels = list(r["cand_label"])
        if not cand_ids:
            continue
        pos = [i for i, l in enumerate(labels) if l == 1]
        neg = [i for i, l in enumerate(labels) if l == 0]
        if not pos or not neg:
            continue

        # For each positive, sample negatives
        for pi in pos:
            pos_id = cand_ids[pi]
            neg_idx = np.random.choice(
                neg, size=min(neg_per_pos, len(neg)), replace=False
            )
            for j in [pi] + neg_idx.tolist():
                nid = cand_ids[j]
                lab = int(labels[j])
                meta = news_lookup.get(
                    nid, {"news_idx": 0, "cat_idx": 0, "subcat_idx": 0}
                )
                clicks = int(item_clicks_train.get(nid, 0))
                is_new = 1 if clicks < min_item_train_clicks_for_warm else 0
                rows.append(
                    {
                        "user_id": user_id,
                        "news_id": nid,
                        "user_idx": user_idx,
                        "news_idx": int(meta["news_idx"]),
                        "cat_idx": int(meta["cat_idx"]),
                        "subcat_idx": int(meta["subcat_idx"]),
                        "history_len": float(len(hist)),
                        "item_clicks": float(clicks),
                        "item_clicks_log1p": float(np.log1p(clicks)),
                        "label": lab,
                        "is_cold_user": cold_u,
                        "is_new_item": is_new,
                    }
                )
    return pd.DataFrame(rows)


def build_impressions_for_eval(
    beh: pd.DataFrame,
    news_idx_df: pd.DataFrame,
    maps: IdMaps,
    item_clicks_train: dict[str, int],
    min_user_hist_for_warm: int,
    min_item_train_clicks_for_warm: int,
) -> pd.DataFrame:
    news_lookup = news_idx_df.set_index("news_id")[
        ["news_idx", "cat_idx", "subcat_idx"]
    ].to_dict(orient="index")
    rows = []
    for _, r in tqdm(beh.iterrows(), total=len(beh), desc="Build eval impressions"):
        user_id = str(r["user_id"])
        user_idx = maps.user2idx.get(user_id, 0)
        hist = r["history"]
        cold_u = 1 if is_cold_user(hist, min_user_hist_for_warm) else 0

        cand_ids = list(r["cand_news_id"])
        labels = list(r["cand_label"])
        if not cand_ids:
            continue
        if sum(labels) == 0:
            continue

        cand_news_idx = []
        cand_cat_idx = []
        cand_subcat_idx = []
        cand_is_new = []
        cand_clicks_log1p = []
        for nid in cand_ids:
            meta = news_lookup.get(nid, {"news_idx": 0, "cat_idx": 0, "subcat_idx": 0})
            clicks = int(item_clicks_train.get(nid, 0))
            is_new = 1 if clicks < min_item_train_clicks_for_warm else 0
            cand_news_idx.append(int(meta["news_idx"]))
            cand_cat_idx.append(int(meta["cat_idx"]))
            cand_subcat_idx.append(int(meta["subcat_idx"]))
            cand_is_new.append(is_new)
            cand_clicks_log1p.append(float(np.log1p(clicks)))

        rows.append(
            {
                "impression_id": str(r["impression_id"]),
                "user_id": user_id,
                "user_idx": user_idx,
                "history_len": float(len(hist)),
                "is_cold_user": cold_u,
                "cand_news_id": cand_ids,
                "cand_label": labels,
                "cand_news_idx": cand_news_idx,
                "cand_cat_idx": cand_cat_idx,
                "cand_subcat_idx": cand_subcat_idx,
                "cand_is_new_item": cand_is_new,
                "cand_item_clicks_log1p": cand_clicks_log1p,
            }
        )
    return pd.DataFrame(rows)


def run_preprocess(cfg: dict[str, Any]) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

    raw_root = Path(cfg["data"]["raw_root"])
    ds = cfg["data"]["dataset_name"]
    train_dir = raw_root / cfg["data"]["train_dir"]
    dev_dir = raw_root / cfg["data"]["dev_dir"]

    news_train = read_news_tsv(train_dir / "news.tsv")
    beh_train = read_behaviors_tsv(train_dir / "behaviors.tsv")

    news_dev = read_news_tsv(dev_dir / "news.tsv")
    beh_dev = read_behaviors_tsv(dev_dir / "behaviors.tsv")

    # Optionally sub-sample behaviors (faster laptop iteration)
    ss = cfg["data"].get("sub_sample", {})
    if ss.get("enabled", False):
        beh_train = sub_sample_behaviors(beh_train, int(ss["train_impressions"]), seed)
        beh_dev = sub_sample_behaviors(beh_dev, int(ss["dev_impressions"]), seed)

    news_all = (
        pd.concat([news_train, news_dev], axis=0)
        .drop_duplicates("news_id")
        .reset_index(drop=True)
    )

    maps = build_id_maps(news_all, beh_train)
    proc_root = ensure_dir(Path(cfg["data"]["processed_root"]) / ds)
    maps_path = proc_root / "id_maps.json"
    maps.save(maps_path)

    news_idx_df = add_indices(news_all, maps)
    news_idx_df.to_parquet(proc_root / "news.parquet", index=False)

    # Item clicks in train from impressions
    click_counts = {}
    for ids, labs in zip(beh_train["cand_news_id"], beh_train["cand_label"]):
        for nid, lab in zip(ids, labs):
            if lab == 1:
                click_counts[nid] = click_counts.get(nid, 0) + 1
    save_json(proc_root / "item_click_counts.json", click_counts)

    pairs_train = build_pairs(
        beh=beh_train,
        news_idx_df=news_idx_df,
        maps=maps,
        item_clicks_train=click_counts,
        min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
        min_item_train_clicks_for_warm=int(
            cfg["data"]["min_item_train_clicks_for_warm"]
        ),
        neg_per_pos=4,
        seed=seed,
    )
    pairs_dev = build_pairs(
        beh=beh_dev,
        news_idx_df=news_idx_df,
        maps=maps,
        item_clicks_train=click_counts,
        min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
        min_item_train_clicks_for_warm=int(
            cfg["data"]["min_item_train_clicks_for_warm"]
        ),
        neg_per_pos=4,
        seed=seed + 1,
    )

    pairs_train.to_parquet(proc_root / "train_pairs.parquet", index=False)
    pairs_dev.to_parquet(proc_root / "dev_pairs.parquet", index=False)

    impr_dev = build_impressions_for_eval(
        beh=beh_dev,
        news_idx_df=news_idx_df,
        maps=maps,
        item_clicks_train=click_counts,
        min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
        min_item_train_clicks_for_warm=int(
            cfg["data"]["min_item_train_clicks_for_warm"]
        ),
    )
    impr_dev.to_parquet(proc_root / "dev_impressions.parquet", index=False)

    meta = {
        "dataset": ds,
        "n_news": int(len(news_idx_df)),
        "n_train_impressions": int(len(beh_train)),
        "n_dev_impressions": int(len(beh_dev)),
        "n_train_pairs": int(len(pairs_train)),
        "n_dev_pairs": int(len(pairs_dev)),
        "n_dev_eval_impressions": int(len(impr_dev)),
    }
    save_json(proc_root / "preprocess_meta.json", meta)
