"""Receiver-scene expression error and recovery accumulation."""

import torch

from phygrec.transforms import normalized_log


class ExpressionMetricAccumulator:
    """Accumulate float64 sums after the original float32 per-entry errors."""

    def __init__(self) -> None:
        self.entries = 0
        self.genes = 0
        self.base_log_abs = 0.0
        self.pred_log_abs = 0.0
        self.base_relative = 0.0
        self.pred_relative = 0.0
        self.base_count_abs = 0.0
        self.pred_count_abs = 0.0
        self.pred_log_square = 0.0
        self.pred_count_square = 0.0
        self.pred_relative_l2 = 0.0

    def update(self, initial: torch.Tensor, prediction: torch.Tensor,
               clean: torch.Tensor) -> None:
        count, genes = clean.shape
        if self.genes not in (0, genes):
            raise ValueError("Gene count changed during evaluation")
        self.genes = genes
        self.entries += count
        clean_log = normalized_log(clean)
        base_log_error = normalized_log(initial) - clean_log
        pred_log_error = normalized_log(prediction) - clean_log
        base_count_error = initial - clean
        pred_count_error = prediction - clean
        self.base_log_abs += float(base_log_error.double().abs().sum().cpu())
        self.pred_log_abs += float(pred_log_error.double().abs().sum().cpu())
        total = clean.sum(dim=1).clamp_min(1.0)
        self.base_relative += float((base_count_error.abs().sum(dim=1) / total).double().sum().cpu())
        self.pred_relative += float((pred_count_error.abs().sum(dim=1) / total).double().sum().cpu())
        self.base_count_abs += float(base_count_error.double().abs().sum().cpu())
        self.pred_count_abs += float(pred_count_error.double().abs().sum().cpu())
        self.pred_log_square += float(pred_log_error.double().square().sum().cpu())
        self.pred_count_square += float(pred_count_error.double().square().sum().cpu())
        reference_norm = torch.linalg.vector_norm(clean, dim=1).clamp_min(1.0)
        self.pred_relative_l2 += float(
            (torch.linalg.vector_norm(pred_count_error, dim=1) / reference_norm).double().sum().cpu()
        )

    def summary(self) -> dict:
        if not self.entries:
            raise ValueError("Empty evaluation split")
        elements = self.entries * self.genes
        input_log_mae = self.base_log_abs / elements
        pred_log_mae = self.pred_log_abs / elements
        input_relative = self.base_relative / self.entries
        pred_relative = self.pred_relative / self.entries
        input_count_mae = self.base_count_abs / elements
        pred_count_mae = self.pred_count_abs / elements
        profile_recovery = 1 - pred_log_mae / max(input_log_mae, 1e-8)
        raw_recovery = 1 - pred_relative / max(input_relative, 1e-8)
        return {
            "receiver_scene_entries": self.entries,
            "genes": self.genes,
            "input_profile_mae": input_log_mae,
            "pred_profile_mae": pred_log_mae,
            "input_raw_l1": input_relative,
            "pred_raw_l1": pred_relative,
            "input_raw_mae": input_count_mae,
            "pred_raw_mae": pred_count_mae,
            "profile_recovery": profile_recovery,
            "raw_recovery": raw_recovery,
            "combined_recovery": (profile_recovery + raw_recovery) / 2,
            "count_rmse": (self.pred_count_square / elements) ** 0.5,
            "log_rmse": (self.pred_log_square / elements) ** 0.5,
            "relative_l2": self.pred_relative_l2 / self.entries,
        }
