"""Circle-overlap observation operator and sparse graph prediction."""

from __future__ import annotations

import torch
from torch import nn

from phygrec.transforms import _inverse_softplus_scalar


def graph_prediction(
    source_profile: torch.Tensor,
    source_index: torch.Tensor,
    donor_index: torch.Tensor,
    donor_fraction: torch.Tensor,
    donor_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct mixed observations from source profiles and donor contributions.

    This is the fixed forward model X = AS: each observation is the sum of its
    source cell's expression plus weighted donor contributions.
    """
    prediction = source_profile[source_index].clone()
    for slot in range(donor_index.shape[1]):
        mask = donor_mask[:, slot]
        if torch.any(mask):
            prediction[mask] += (
                source_profile[donor_index[mask, slot]]
                * donor_fraction[mask, slot].unsqueeze(-1)
            )
    return prediction


def _intersection_fraction(
    distance: torch.Tensor,
    receiver_radius: torch.Tensor,
    donor_radius: torch.Tensor,
) -> torch.Tensor:
    safe_distance = distance.clamp_min(1e-3)
    epsilon = torch.finfo(distance.dtype).eps * 32
    receiver_squared = receiver_radius.square()
    donor_squared = donor_radius.square()
    receiver_angle = (
        (safe_distance.square() + receiver_squared - donor_squared)
        / (2.0 * safe_distance * receiver_radius).clamp_min(epsilon)
    ).clamp(-1.0 + epsilon, 1.0 - epsilon)
    donor_angle = (
        (safe_distance.square() + donor_squared - receiver_squared)
        / (2.0 * safe_distance * donor_radius).clamp_min(epsilon)
    ).clamp(-1.0 + epsilon, 1.0 - epsilon)
    radicand = (
        (-safe_distance + receiver_radius + donor_radius)
        * (safe_distance + receiver_radius - donor_radius)
        * (safe_distance - receiver_radius + donor_radius)
        * (safe_distance + receiver_radius + donor_radius)
    )
    lens_area = (
        receiver_squared * torch.acos(receiver_angle)
        + donor_squared * torch.acos(donor_angle)
        - 0.5 * radicand.clamp_min(epsilon).sqrt()
    )
    contained_area = torch.pi * torch.minimum(receiver_radius, donor_radius).square()
    area = torch.where(
        safe_distance >= receiver_radius + donor_radius,
        torch.zeros_like(lens_area),
        torch.where(
            safe_distance <= torch.abs(receiver_radius - donor_radius),
            contained_area,
            lens_area,
        ),
    )
    return area / (torch.pi * donor_squared).clamp_min(epsilon)


class CircleOverlapOperator(nn.Module):
    """An independent four-parameter circle operator for each inverse Block."""

    def __init__(self) -> None:
        super().__init__()
        self.iterations = 4
        self.log_base_radius = nn.Parameter(torch.full(
            (self.iterations,), _inverse_softplus_scalar(50.0), dtype=torch.float32
        ))
        self.total_exponent_logit = nn.Parameter(torch.full(
            (self.iterations,), -2.0, dtype=torch.float32
        ))
        self.receiver_scale_log = nn.Parameter(torch.full(
            (self.iterations,), _inverse_softplus_scalar(1.0), dtype=torch.float32
        ))
        self.efficiency_logit = nn.Parameter(torch.full(
            (self.iterations,), 2.2, dtype=torch.float32
        ))

    def physical_parameters_at(self, iteration: int) -> dict[str, torch.Tensor]:
        if not 0 <= int(iteration) < self.iterations:
            raise ValueError(
                f"circle iteration must be in [0, {self.iterations}), got {iteration}"
            )
        return {
            "base_radius": nn.functional.softplus(self.log_base_radius[iteration]),
            "total_exponent": nn.functional.softplus(self.total_exponent_logit[iteration]),
            "receiver_scale": nn.functional.softplus(self.receiver_scale_log[iteration]),
            "transfer_efficiency": torch.sigmoid(self.efficiency_logit[iteration]),
        }

    def forward(
        self,
        distance: torch.Tensor,
        candidate_mask: torch.Tensor,
        receiver_total: torch.Tensor,
        donor_total: torch.Tensor,
        *,
        iteration: int,
    ) -> torch.Tensor:
        if distance.shape != candidate_mask.shape or donor_total.shape != distance.shape:
            raise ValueError("candidate edge tensor shapes must match")
        if receiver_total.shape != distance.shape[:1]:
            raise ValueError("receiver_total must have one value per observation")
        effective_mask = candidate_mask & torch.isfinite(distance)
        if not torch.any(effective_mask):
            raise ValueError("candidate graph has no finite spatial distances")
        finite_distance = torch.where(
            effective_mask, distance.clamp_min(0.0), torch.zeros_like(distance)
        )
        positive_total = torch.cat([receiver_total, donor_total[effective_mask]])
        positive_total = positive_total[positive_total > 0]
        if not torch.numel(positive_total):
            raise ValueError("expression totals must contain a positive value")
        total_scale = positive_total.detach().median().clamp_min(1.0)
        parameters = self.physical_parameters_at(iteration)
        receiver_radius = parameters["base_radius"] * (
            receiver_total.clamp_min(1.0) / total_scale
        ).pow(parameters["total_exponent"])
        receiver_radius = receiver_radius * parameters["receiver_scale"]
        donor_radius = parameters["base_radius"] * (
            donor_total.clamp_min(1.0) / total_scale
        ).pow(parameters["total_exponent"])
        overlap = _intersection_fraction(
            finite_distance,
            receiver_radius.unsqueeze(-1),
            donor_radius,
        )
        raw = parameters["transfer_efficiency"] * overlap * effective_mask.float()
        return raw
