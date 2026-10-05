"""Graph correction of each physical inverse increment."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch_geometric.nn as pyg_nn


def bidirectional_candidate_edge_index(
    source: torch.Tensor,
    donors: torch.Tensor,
    donor_mask: torch.Tensor,
) -> torch.Tensor:
    """Return directed donor/receiver edges without crossing packed graphs.

    Packed scene indices are already offset by the data loader.  Deriving
    edges exclusively from those source/donor pairs therefore preserves graph
    separation without needing graph identifiers or CPU-side set operations.
    Duplicate observations are harmless for attention and retain their
    multiplicity as observable evidence.
    """

    if donors.shape != donor_mask.shape:
        raise ValueError("candidate donors and mask shapes must match")
    rows, slots = donor_mask.bool().nonzero(as_tuple=True)
    if rows.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=source.device)
    sender = donors[rows, slots]
    receiver = source[rows]
    if torch.any(sender < 0):
        raise ValueError("active candidate donor index must be non-negative")
    keep = sender != receiver
    sender = sender[keep]
    receiver = receiver[keep]
    if sender.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=source.device)
    forward = torch.stack((sender, receiver), dim=0)
    reverse = torch.stack((receiver, sender), dim=0)
    return torch.cat((forward, reverse), dim=1)


class GraphContextCorrector(nn.Module):
    """Learn a graph correction from the current state and physical update."""

    def __init__(self) -> None:
        super().__init__()
        n_genes, hidden_dim, hidden_layers, heads = 1000, 128, 2, 4
        initial_strength, dropout = 0.1, 0.0
        self.n_genes = n_genes

        self.state_norm = nn.LayerNorm(self.n_genes)
        self.update_norm = nn.LayerNorm(self.n_genes)
        self.input_projection = nn.Linear(2 * self.n_genes, hidden_dim)
        self.convs = nn.ModuleList(
            pyg_nn.GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=int(heads),
                concat=False,
                dropout=float(dropout),
                add_self_loops=True,
            )
            for _ in range(hidden_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(hidden_layers)
        )
        self.output = nn.Linear(hidden_dim, self.n_genes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        initial_logit = float(
            initial_strength + math.log(-math.expm1(-initial_strength))
        )
        self.strength_logits = nn.Parameter(
            torch.full((1,), initial_logit)
        )

    def forward(
        self,
        profile: torch.Tensor,
        physical_update: torch.Tensor,
        source: torch.Tensor,
        donors: torch.Tensor,
        donor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a signed correction for the physical increment."""

        if profile.shape != physical_update.shape:
            raise ValueError("profile and physical update shapes must match")
        if profile.ndim != 2 or profile.shape[1] != self.n_genes:
            raise ValueError(
                f"graph corrector expected {self.n_genes} genes, "
                f"received shape {tuple(profile.shape)}"
            )
        # Express the signed physical update relative to an observable local
        # count scale.  asinh is linear near zero and logarithmic for outliers.
        cell_unit = profile.detach().mean(dim=1, keepdim=True).clamp_min(1e-6)
        state_features = torch.log1p(profile.clamp_min(0.0))
        update_features = torch.asinh(physical_update / cell_unit)
        features = torch.cat(
            (self.state_norm(state_features), self.update_norm(update_features)),
            dim=-1,
        )
        hidden = torch.nn.functional.gelu(self.input_projection(features))
        edge_index = bidirectional_candidate_edge_index(
            source, donors, donor_mask
        )
        for convolution, normalization in zip(self.convs, self.norms):
            message = convolution(hidden, edge_index)
            hidden = normalization(hidden + torch.nn.functional.gelu(message))
        raw_output = self.output(hidden)
        return torch.nn.functional.softplus(self.strength_logits[0]) * cell_unit * raw_output
