"""Shared file hashing and numerical transforms."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inverse_softplus_scalar(value: float) -> float:
    """Return a numerically stable scalar whose softplus is ``value``."""
    if value <= 0.0:
        raise ValueError("inverse softplus requires a positive value")
    return float(value + math.log(-math.expm1(-value)))


def normalized_log(raw: torch.Tensor) -> torch.Tensor:
    """Library-size normalised log1p transform for profile MAE computation."""
    import torch

    raw = raw.clamp_min(0.0)
    total = raw.sum(dim=1).clamp_min(1e-6)
    return torch.log1p(raw * (1e4 / total).unsqueeze(-1))
