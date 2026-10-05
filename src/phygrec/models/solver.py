"""Four-Block nonnegative recovery with RB, AIM and GCC branches."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from phygrec.models.modulation import AdaptiveIncrementModulation
from phygrec.models.operators import CircleOverlapOperator, graph_prediction
from phygrec.protocol import ABLATIONS, Ablation
from phygrec.transforms import _inverse_softplus_scalar


class UnrolledRecoverySolver(nn.Module):
    """Four fixed PGR Blocks, with exactly one published branch removal."""

    def __init__(self, ablation: Ablation = 'none') -> None:
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f'Unknown ablation: {ablation}')
        self.ablation = ablation
        iterations = self.iterations = 4
        robust_threshold_init, robust_barron_shape_init = 1.0, 1.0

        self.step_logits = nn.Parameter(
            torch.full((iterations,), _inverse_softplus_scalar(0.2))
        )
        initial_momentum = 0.5 / (1.0 + math.exp(3.0))
        initial_momentum_logit = math.log(initial_momentum / (1.0 - initial_momentum))
        if ablation != "aim":
            self.momentum_logits = nn.Parameter(
                torch.full((iterations,), initial_momentum_logit)
            )
        else:
            # The published AIM removal disables gain and momentum together.
            self.register_parameter("momentum_logits", None)
        threshold_logit = torch.log(torch.expm1(torch.tensor(robust_threshold_init)))
        self.robust_threshold_logits = nn.Parameter(threshold_logit.repeat(iterations))
        shape_logit = torch.tensor(
            _inverse_softplus_scalar(2.0 - float(robust_barron_shape_init))
        )
        self.robust_shape_logits = nn.Parameter(shape_logit.repeat(iterations))
        initial_layerscale = torch.tensor(_inverse_softplus_scalar(1.0))
        self.iteration_layerscale_logits = nn.Parameter(
            initial_layerscale.repeat(iterations)
        )
        self.adaptive_modulators = nn.ModuleList()

        if ablation != "aim":
            self.adaptive_modulators = nn.ModuleList(
                AdaptiveIncrementModulation()
                for _ in range(self.iterations)
            )
        self.graph_correctors = nn.ModuleList()
        if ablation != "gcc":
            from phygrec.models.graph_corrector import GraphContextCorrector

            self.graph_correctors = nn.ModuleList(
                GraphContextCorrector()
                for _ in range(self.iterations)
            )

    def iteration_layerscales(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.iteration_layerscale_logits)

    def step_sizes(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.step_logits)

    def momenta(self) -> torch.Tensor:
        if self.momentum_logits is None:
            return self.step_logits.new_zeros((self.iterations,))
        return torch.sigmoid(self.momentum_logits)

    def robust_shapes(self) -> torch.Tensor:
        return 2.0 - torch.nn.functional.softplus(self.robust_shape_logits)

    def forward(
        self,
        initial: torch.Tensor,
        mixed: torch.Tensor,
        source: torch.Tensor,
        donors: torch.Tensor,
        donor_mask: torch.Tensor,
        iteration_fractions: torch.Tensor,
    ) -> torch.Tensor:
        def physical_degree(current_fraction: torch.Tensor) -> torch.Tensor:
            value = torch.zeros(len(initial), device=initial.device)
            value.index_add_(
                0, source, torch.ones_like(source, dtype=torch.float32)
            )
            for slot in range(donors.shape[1]):
                mask = donor_mask[:, slot]
                if torch.any(mask):
                    value.index_add_(
                        0,
                        donors[mask, slot],
                        current_fraction[mask, slot].square(),
                    )
            return value.clamp_min(1e-6).unsqueeze(-1)

        if iteration_fractions.shape != (self.iterations, *donors.shape):
            raise ValueError("iteration_fractions must match Blocks and candidate edges")
        profile = initial
        previous_update = torch.zeros_like(profile)

        for iteration in range(self.iterations):
            fraction = iteration_fractions[iteration]
            degree = physical_degree(fraction)
            prediction = graph_prediction(profile, source, donors, fraction, donor_mask)
            residual = prediction - mixed
            relative_residual = (
                residual.abs().sum(dim=1) / mixed.abs().sum(dim=1).clamp_min(1.0)
            )
            threshold = torch.nn.functional.softplus(
                self.robust_threshold_logits[iteration]
            ).clamp_min(1e-4)
            standardized_squared = (relative_residual / threshold).square()
            shape = self.robust_shapes()[iteration]
            distance_from_two = (shape - 2.0).abs().clamp_min(1e-4)
            observation_weight = (
                1.0 + standardized_squared / distance_from_two
            ).pow(0.5 * shape - 1.0)
            effective_residual = residual * observation_weight.unsqueeze(-1)

            gradient = torch.zeros_like(profile)
            gradient.index_add_(0, source, effective_residual)
            for slot in range(donors.shape[1]):
                mask = donor_mask[:, slot]
                if torch.any(mask):
                    gradient.index_add_(
                        0,
                        donors[mask, slot],
                        effective_residual[mask] * fraction[mask, slot].unsqueeze(-1),
                    )

            step = self.step_sizes()[iteration]
            momentum = (
                0.0 if self.momentum_logits is None else self.momenta()[iteration]
            )
            gradient_update = -step * gradient / degree
            gradient_update = self.iteration_layerscales()[iteration] * gradient_update
            if self.ablation == "rb":
                # Exact zero at the (a)->(b) boundary, on every block.
                # Keep the circle operator, graph and momentum path intact.
                gradient_update = torch.zeros_like(gradient_update)
            active_modulator = (
                self.adaptive_modulators[iteration]
                if self.adaptive_modulators else None
            )
            if active_modulator is not None:
                gain = active_modulator.gain(
                    profile,
                    physical_gradient_update=gradient_update,
                    iteration=iteration,
                )
                gradient_update = gradient_update * gain
            # The gain acts on the physical gradient before explicit momentum.
            update = gradient_update + momentum * previous_update
            active_graph_corrector = (
                self.graph_correctors[iteration] if self.graph_correctors else None
            )
            if active_graph_corrector is not None:
                graph_correction = active_graph_corrector(
                    profile,
                    physical_update=update,
                    source=source,
                    donors=donors,
                    donor_mask=donor_mask,
                )
                unconstrained_proposal = profile + update + graph_correction
            else:
                unconstrained_proposal = profile + update
            proposal = torch.relu(unconstrained_proposal)
            previous_profile = profile
            profile = profile + (proposal - profile)
            previous_update = profile - previous_profile

        return profile


class PhyGRecSolver(nn.Module):
    def __init__(self, ablation: Ablation = 'none') -> None:
        super().__init__()
        self.forward_operator = CircleOverlapOperator()
        self.solver = UnrolledRecoverySolver(ablation)
        self.output_residual_logit = nn.Parameter(torch.tensor(0.0))

    def output_residual_scale(self) -> torch.Tensor:
        return torch.exp(self.output_residual_logit)

    def _transfer_fraction(self, graph: dict, iteration: int) -> torch.Tensor:
        expression = graph["initial"]
        total = expression.sum(dim=1)
        return self.forward_operator(
            graph["distance"],
            graph["candidate_mask"],
            total[graph["source"]],
            total[graph["donors"].clamp_min(0)],
            iteration=iteration,
        )

    def forward(self, graph: dict) -> torch.Tensor:
        iteration_fractions = torch.stack(
            [
                self._transfer_fraction(graph, iteration)
                for iteration in range(self.forward_operator.iterations)
            ],
            dim=0,
        )
        initial = graph["initial"]
        mixed = initial[graph["source"]]

        solver_inputs = (
            initial, mixed, graph["source"], graph["donors"],
            graph["candidate_mask"], iteration_fractions,
        )
        if self.training and torch.is_grad_enabled():
            prediction = checkpoint(
                self.solver, *solver_inputs, use_reentrant=False
            )
        else:
            prediction = self.solver(*solver_inputs)
        prediction = torch.clamp_min(
            initial + self.output_residual_scale() * (prediction - initial), 0.0
        )
        return prediction
