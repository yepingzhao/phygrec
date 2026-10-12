"""Expose only model-visible candidate-graph inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_MODEL_KEYS = frozenset(
    {"initial", "source", "donors", "distance", "candidate_mask"}
)


def build_compliant_model_input(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only the fixed observable whitelist from a full scene graph."""
    missing = REQUIRED_MODEL_KEYS - set(graph)
    if missing:
        raise KeyError(f"candidate graph lacks model-visible keys {sorted(missing)}")
    return {key: graph[key] for key in REQUIRED_MODEL_KEYS}
