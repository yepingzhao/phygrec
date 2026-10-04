"""Adaptive increment modulation for the four unrolled Blocks."""

from __future__ import annotations

import torch
from torch import nn

from phygrec.transforms import _inverse_softplus_scalar


class AdaptiveGainNetwork(nn.Module):
    """Block-specific expression-conditioned gain coordinates."""

    def __init__(self) -> None:
        super().__init__()
        n_genes, hidden_dim, iterations = 1000, 16, 4
        self.n_genes = n_genes
        self.iterations = iterations
        self.norm = nn.RMSNorm(n_genes)
        self.down = nn.Linear(n_genes, 2 * hidden_dim)
        self.update_norm = nn.RMSNorm(n_genes)
        self.step_embedding = nn.Embedding(iterations, 8)
        self.step_in = nn.Linear(8, hidden_dim)
        self.step_out = nn.Linear(hidden_dim, hidden_dim)
        self.concat_project = nn.Linear(3 * hidden_dim, 2 * hidden_dim)
        with torch.no_grad():
            self.concat_project.weight.zero_()
            self.concat_project.bias.zero_()
            self.concat_project.weight[:, : 2 * hidden_dim].copy_(
                torch.eye(2 * hidden_dim)
            )
            nn.init.xavier_uniform_(
                self.concat_project.weight[:, 2 * hidden_dim :]
            )
        nn.init.zeros_(self.step_out.weight)
        nn.init.zeros_(self.step_out.bias)
        self.up = nn.Linear(hidden_dim, n_genes)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        expression: torch.Tensor,
        iteration: int,
        physical_update: torch.Tensor,
    ) -> torch.Tensor:
        if expression.ndim != 2 or expression.shape[1] != self.n_genes:
            raise ValueError("expression has the wrong gene dimension")
        if not 0 <= int(iteration) < self.iterations:
            raise ValueError("iteration is outside the inverse Blocks")
        if physical_update.shape != expression.shape:
            raise ValueError("physical_update must match expression")
        network_input = self.norm(expression) + self.update_norm(physical_update)
        hidden = self.down(network_input)
        index = torch.tensor(iteration, device=expression.device, dtype=torch.long)
        embedding = self.step_embedding(index)
        injection = self.step_out(torch.nn.functional.silu(self.step_in(embedding)))
        injection = injection.unsqueeze(0).expand(hidden.shape[0], -1)
        hidden = self.concat_project(torch.cat((hidden, injection), dim=-1))
        value, gate = hidden.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(value) * gate
        return self.up(hidden)


class AdaptiveIncrementModulation(nn.Module):
    """Positive learned gain applied to the physical gradient before momentum."""

    def __init__(self) -> None:
        super().__init__()
        self.gain_network = AdaptiveGainNetwork()

    def gain(
        self,
        expression: torch.Tensor,
        physical_gradient_update: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        coordinate = self.gain_network(
            expression,
            iteration=iteration,
            physical_update=physical_gradient_update,
        )
        offset = coordinate.new_tensor(_inverse_softplus_scalar(1.0))
        return torch.nn.functional.softplus(coordinate + offset)
