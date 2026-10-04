"""Scalar parameterization and expression normalization."""

from __future__ import annotations

import math

import torch


def _inverse_softplus_scalar(value: float) -> float:
    """Return a numerically stable scalar whose softplus is ``value``."""
    if value <= 0.0:
        raise ValueError("inverse softplus requires a positive value")
    return float(value + math.log(-math.expm1(-value)))


def normalized_log(raw: torch.Tensor) -> torch.Tensor:
    """Library-size normalised log1p transform for profile MAE computation."""
    raw = raw.clamp_min(0.0)
    total = raw.sum(dim=1).clamp_min(1e-6)
    return torch.log1p(raw * (1e4 / total).unsqueeze(-1))
