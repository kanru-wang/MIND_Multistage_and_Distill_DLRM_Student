from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_distill_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    t = float(temperature)
    ps = F.log_softmax(student_logits / t, dim=0)
    pt = F.softmax(teacher_logits / t, dim=0)
    return F.kl_div(ps, pt, reduction="batchmean") * (t * t)


def pairwise_logit_distill_bce(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    # For pairwise samples, distill teacher probability sigmoid(logit/T)
    t = float(temperature)
    target = torch.sigmoid(teacher_logits / t)
    return F.binary_cross_entropy_with_logits(student_logits, target)


def repr_distill_mse(
    student_repr: torch.Tensor, teacher_repr: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(student_repr, teacher_repr)
