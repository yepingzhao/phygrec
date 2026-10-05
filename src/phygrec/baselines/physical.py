"""Frozen circle geometry and fixed projected-gradient inversion."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import nn
from phygrec.models.operators import _intersection_fraction, graph_prediction


class FittedCircleOperator(nn.Module):
    """Four-parameter approximation to contour-overlap crosstalk.

    Observable initial-profile totals act as a proxy for cell radius.
    The edge fraction is the intersection area of two circles divided by the
    donor-circle area, matching the overlap convention used by the simulator.
    """

    def __init__(self) -> None:
        maximum_row_crosstalk = 0.8
        super().__init__()
        self.maximum_row_crosstalk = float(maximum_row_crosstalk)
        initial_radius_raw = float(np.log(50.0))
        initial_receiver_scale_raw = 0.0
        self.log_base_radius = nn.Parameter(torch.tensor(initial_radius_raw, dtype=torch.float32))
        self.total_exponent_logit = nn.Parameter(torch.tensor(-2.0))
        self.receiver_scale_log = nn.Parameter(torch.tensor(initial_receiver_scale_raw))
        self.efficiency_logit = nn.Parameter(torch.tensor(2.2))

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        radius_raw = self.log_base_radius
        receiver_scale_raw = self.receiver_scale_log
        return {'base_radius': torch.exp(radius_raw.clamp(-4.0, 8.0)), 'total_exponent': nn.functional.softplus(self.total_exponent_logit), 'receiver_scale': torch.exp(receiver_scale_raw.clamp(-3.0, 3.0)), 'transfer_efficiency': torch.sigmoid(self.efficiency_logit)}

    @staticmethod
    def _intersection_fraction(distance: torch.Tensor, receiver_radius: torch.Tensor, donor_radius: torch.Tensor) -> torch.Tensor:
        return _intersection_fraction(distance, receiver_radius, donor_radius)

    def forward(self, distance: torch.Tensor, candidate_mask: torch.Tensor, receiver_total: torch.Tensor, donor_total: torch.Tensor, observation_graph_index: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if distance.shape != candidate_mask.shape or donor_total.shape != distance.shape:
            raise ValueError('candidate edge tensor shapes must match')
        if receiver_total.shape != distance.shape[:1]:
            raise ValueError('receiver_total must have one value per observation')
        effective_mask = candidate_mask & torch.isfinite(distance)
        if not torch.any(effective_mask):
            raise ValueError('candidate graph has no finite spatial distances')
        finite_distance = torch.where(effective_mask, distance.clamp_min(0.0), torch.zeros_like(distance))
        if observation_graph_index is None:
            positive_total = torch.cat([receiver_total, donor_total[effective_mask]])
            positive_total = positive_total[positive_total > 0]
            if not torch.numel(positive_total):
                raise ValueError('expression totals must contain a positive value')
            total_scale = positive_total.detach().median().clamp_min(1.0)
        else:
            if observation_graph_index.shape != receiver_total.shape:
                raise ValueError('one graph index is required per observation')
            scales = []
            for graph_id in range(int(observation_graph_index.max().item()) + 1):
                rows = observation_graph_index == graph_id
                values = torch.cat([receiver_total[rows], donor_total[rows][effective_mask[rows]]])
                values = values[values > 0]
                if not torch.numel(values):
                    raise ValueError(f'packed graph {graph_id} has no positive expression total')
                scales.append(values.detach().median().clamp_min(1.0))
            total_scale = torch.stack(scales)[observation_graph_index]
        parameters = self.physical_parameters()
        receiver_radius = (parameters['base_radius'] * (receiver_total.clamp_min(1.0) / total_scale).pow(parameters['total_exponent']) * parameters['receiver_scale'])[:, None]
        donor_scale = total_scale if total_scale.ndim == 0 else total_scale[:, None]
        donor_radius = parameters['base_radius'] * (donor_total.clamp_min(1.0) / donor_scale).pow(parameters['total_exponent'])
        fraction = parameters['transfer_efficiency'] * self._intersection_fraction(finite_distance, receiver_radius, donor_radius)
        fraction = torch.where(effective_mask, fraction, torch.zeros_like(fraction))
        row_total = fraction.sum(dim=1)
        row_scale = torch.clamp(self.maximum_row_crosstalk / row_total.clamp_min(1e-08), max=1.0)
        fraction = fraction * row_scale[:, None]
        return (fraction, fraction.sum(dim=1))

def solve_physical_pgd(batch: dict[str, Any], fraction: torch.Tensor, step: float, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
    source, donors, mask = (batch['source'], batch['donors'], batch['candidate_mask'])
    profile = batch['initial'].clone()
    degree = torch.zeros(len(profile), device=profile.device)
    degree.index_add_(0, source, torch.ones_like(source, dtype=torch.float32))
    for slot in range(donors.shape[1]):
        m = mask[:, slot]
        if m.any():
            degree.index_add_(0, donors[m, slot], fraction[m, slot].square())
    degree = degree.clamp_min(1e-06).unsqueeze(-1)
    for _ in range(iterations):
        pred = graph_prediction(profile, source, donors, fraction, mask)
        residual = pred - batch['mixed']
        grad = torch.zeros_like(profile)
        grad.index_add_(0, source, residual)
        for slot in range(donors.shape[1]):
            m = mask[:, slot]
            if m.any():
                grad.index_add_(0, donors[m, slot], residual[m] * fraction[m, slot].unsqueeze(-1))
        profile = torch.relu(profile - step * grad / degree)
    return (profile, graph_prediction(profile, source, donors, fraction, mask))
