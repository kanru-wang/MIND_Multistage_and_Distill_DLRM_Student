from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mindrec.metrics.calibration import brier_score, expected_calibration_error
from mindrec.utils import load_json, save_json


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        t = max(float(self.temperature), 1.0e-6)
        return np.asarray(logits, dtype=np.float32) / t

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform_logits(logits)
        return 1.0 / (1.0 + np.exp(-scaled))

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        payload = {
            "method": "temperature",
            "temperature": float(self.temperature),
        }
        if meta:
            payload["meta"] = meta
        save_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        payload = load_json(path)
        return cls(temperature=float(payload.get("temperature", 1.0)))


def fit_temperature_scaler(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
    lr: float = 0.05,
) -> tuple[TemperatureScaler, dict[str, float]]:
    x = torch.tensor(np.asarray(logits, dtype=np.float32))
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    log_t = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    opt = torch.optim.Adam([log_t], lr=lr)

    for _ in range(max_iter):
        opt.zero_grad()
        t = torch.exp(log_t).clamp_min(1.0e-6)
        loss = F.binary_cross_entropy_with_logits(x / t, y)
        loss.backward()
        opt.step()

    scaler = TemperatureScaler(temperature=float(torch.exp(log_t).item()))
    raw_prob = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
    cal_prob = scaler.predict_proba(np.asarray(logits, dtype=np.float32))
    y_np = np.asarray(labels, dtype=np.float32)
    stats = {
        "nll_raw": float(
            F.binary_cross_entropy_with_logits(x, y).detach().cpu().item()
        ),
        "nll_calibrated": float(
            F.binary_cross_entropy_with_logits(
                torch.tensor(scaler.transform_logits(logits)), y
            )
            .detach()
            .cpu()
            .item()
        ),
        "brier_raw": brier_score(y_np, raw_prob),
        "brier_calibrated": brier_score(y_np, cal_prob),
        "ece_15_raw": expected_calibration_error(y_np, raw_prob, n_bins=15),
        "ece_15_calibrated": expected_calibration_error(y_np, cal_prob, n_bins=15),
    }
    return scaler, stats
