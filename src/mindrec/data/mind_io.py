from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


NEWS_COLUMNS=[
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


BEH_COLUMNS=[
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


def read_news_tsv(path: str | Path) -> pd.DataFrame:
    df=pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        quoting=3,
        dtype=str,
    )
    for c in ["category","subcategory","title","abstract"]:
        df[c]=df[c].fillna("")
    df["text"]=(df["title"].fillna("") + " [SEP] " + df["abstract"].fillna("")).str.strip()
    return df


def parse_impressions(impr: str) -> tuple[list[str], list[int]]:
    # Format: "N12345-1 N54321-0 ..."
    items=[]
    labels=[]
    if not isinstance(impr, str) or not impr.strip():
        return items, labels
    for tok in impr.strip().split():
        if "-" in tok:
            nid, lab=tok.rsplit("-", 1)
            items.append(nid)
            labels.append(int(lab))
        else:
            items.append(tok)
            labels.append(0)
    return items, labels


def read_behaviors_tsv(path: str | Path) -> pd.DataFrame:
    df=pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BEH_COLUMNS,
        quoting=3,
        dtype=str,
    )
    df["history"]=df["history"].fillna("").apply(lambda s: s.split() if s else [])
    parsed=df["impressions"].fillna("").apply(parse_impressions)
    df["cand_news_id"]=parsed.apply(lambda x: x[0])
    df["cand_label"]=parsed.apply(lambda x: x[1])
    return df


def sub_sample_behaviors(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n<=0 or n>=len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
