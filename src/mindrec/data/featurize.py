from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mindrec.utils import save_json, load_json


@dataclass
class IdMaps:
    user2idx: dict[str, int]
    news2idx: dict[str, int]
    cat2idx: dict[str, int]
    subcat2idx: dict[str, int]

    def save(self, path: str | Path) -> None:
        save_json(path, {
            "user2idx": self.user2idx,
            "news2idx": self.news2idx,
            "cat2idx": self.cat2idx,
            "subcat2idx": self.subcat2idx,
        })

    @staticmethod
    def load(path: str | Path) -> "IdMaps":
        d=load_json(path)
        return IdMaps(
            user2idx=d["user2idx"],
            news2idx=d["news2idx"],
            cat2idx=d["cat2idx"],
            subcat2idx=d["subcat2idx"],
        )


def build_id_maps(news: pd.DataFrame, beh_train: pd.DataFrame) -> IdMaps:
    users=sorted(set(beh_train["user_id"].dropna().astype(str).tolist()))
    news_ids=sorted(set(news["news_id"].dropna().astype(str).tolist()))
    cats=sorted(set(news["category"].fillna("").astype(str).tolist()))
    subcats=sorted(set(news["subcategory"].fillna("").astype(str).tolist()))

    # Reserve 0 for OOV
    user2idx={u:i+1 for i,u in enumerate(users)}
    news2idx={n:i+1 for i,n in enumerate(news_ids)}
    cat2idx={c:i+1 for i,c in enumerate(cats)}
    subcat2idx={s:i+1 for i,s in enumerate(subcats)}
    return IdMaps(user2idx=user2idx, news2idx=news2idx, cat2idx=cat2idx, subcat2idx=subcat2idx)


def add_indices(news: pd.DataFrame, maps: IdMaps) -> pd.DataFrame:
    df=news.copy()
    df["news_idx"]=df["news_id"].map(maps.news2idx).fillna(0).astype(np.int64)
    df["cat_idx"]=df["category"].map(maps.cat2idx).fillna(0).astype(np.int64)
    df["subcat_idx"]=df["subcategory"].map(maps.subcat2idx).fillna(0).astype(np.int64)
    return df


def is_cold_user(history: list[str], min_hist: int) -> bool:
    return len(history) < min_hist
