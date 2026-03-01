from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true=y_true.astype(np.float32)
    y_prob=y_prob.astype(np.float32)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int=15) -> float:
    y_true=y_true.astype(np.float32)
    y_prob=y_prob.astype(np.float32)
    bins=np.linspace(0.0, 1.0, n_bins + 1)
    ece=0.0
    for i in range(n_bins):
        lo,hi=bins[i], bins[i + 1]
        mask=(y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        acc=float(y_true[mask].mean())
        conf=float(y_prob[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)
