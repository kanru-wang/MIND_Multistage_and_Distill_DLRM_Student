from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        return str(o)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def position_bias_weights(k: int, mode: str = "log") -> np.ndarray:
    pos = np.arange(1, k + 1, dtype=np.float32)
    if mode == "log":
        return 1.0 / np.log2(pos + 1.0)
    if mode == "linear":
        return (k - pos + 1.0) / k
    raise ValueError(f"Unknown position bias mode: {mode}")
