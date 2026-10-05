"""The paper objective: raw relative L1 plus normalized-log MAE."""

from __future__ import annotations

import torch

from phygrec.models.solver import PhyGRecSolver
from phygrec.transforms import normalized_log


def recovery_loss(
    model: PhyGRecSolver,
    graph: dict,
    model_graph: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise the final clean expression with raw and profile errors."""
    node_mask = graph["train_mask"]
    prediction = model(model_graph)
    selected_prediction = prediction[node_mask]
    selected_clean = graph["clean"][node_mask]
    total = selected_clean.sum(dim=1).clamp_min(1.0)
    raw_error = (
        (selected_prediction - selected_clean).abs().sum(dim=1) / total
    ).mean()
    profile_per_item = (
        normalized_log(selected_prediction) - normalized_log(selected_clean)
    ).abs().mean(dim=1)
    profile_mae = profile_per_item.mean()
    return raw_error + profile_mae, {
        "clean_raw_l1": raw_error,
        "clean_profile_mae": profile_mae,
    }
