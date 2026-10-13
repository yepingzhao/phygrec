"""Shared file hashing and numerical transforms."""

from __future__ import annotations

import hashlib
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch


@contextmanager
def deterministic_inference():
    """Evaluate in deterministic FP32, then restore the caller's settings."""
    import torch

    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.get_num_threads(), torch.get_float32_matmul_precision(),
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.set_num_threads(previous[2])
        torch.set_float32_matmul_precision(previous[3])
        torch.backends.cudnn.allow_tf32 = previous[4]
        torch.backends.cudnn.deterministic = previous[5]
        torch.backends.cudnn.benchmark = previous[6]
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


@contextmanager
def single_threaded_scoring():
    """Fix BLAS/OpenMP and Numba threads for every scoring entry point."""
    import numba
    from threadpoolctl import threadpool_limits

    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        with threadpool_limits(limits=1):
            yield
    finally:
        numba.set_num_threads(previous)


def package_versions(*names: str) -> dict[str, str]:
    """Record the numerical packages used for an evaluation."""
    from importlib.metadata import version

    return {name: version(name) for name in names}


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


def normalize_log1p(matrix: np.ndarray) -> np.ndarray:
    """Float32 library-size normalization for cell annotation and clustering."""
    import numpy as np

    values = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
    library_size = values.sum(axis=1, keepdims=True)
    scale = np.divide(1e4, library_size, out=np.zeros_like(library_size, dtype=np.float32), where=library_size > 0)
    return np.log1p(values * scale).astype(np.float32, copy=False)
