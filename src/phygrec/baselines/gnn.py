"""The five fixed graph baselines used in the paper."""

from __future__ import annotations
import torch
from torch import nn
import torch_geometric.nn as pyg_nn

def fixed_distance_fraction(distance: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """The fixed geometric prior used by the MPNN baseline.

    Per observation row: ``prior_scale * exp(-delta / (T * scale))`` where
    ``delta`` is distance from the row minimum and ``scale`` is the median of
    the up-to-7 smallest positive deltas (floor 1.0). Zero outside the mask.
    """
    prior_scale, distance_temperature = 0.5, 1.0
    if distance.shape != candidate_mask.shape:
        raise ValueError('distance and candidate_mask shapes must match')
    effective_mask = candidate_mask & torch.isfinite(distance)
    clamped = torch.where(effective_mask, distance.clamp_min(0.0), torch.zeros_like(distance))
    row_min = torch.where(effective_mask, clamped, torch.full_like(clamped, float('inf'))).min(dim=1, keepdim=True).values
    delta = clamped - row_min
    masked_delta = torch.where(effective_mask & (delta > 0.0), delta, torch.full_like(delta, float('inf')))
    sorted_delta, _ = torch.sort(masked_delta, dim=1)
    local = sorted_delta[:, :7]
    finite_count = torch.isfinite(local).sum(dim=1)
    index = torch.arange(local.shape[0], device=distance.device)
    mid = finite_count // 2
    upper = torch.clamp(mid, max=6)
    lower = torch.clamp(mid - 1, min=0, max=6)
    median = torch.where((finite_count % 2 == 1) & (finite_count > 0), local[index, upper], torch.where(finite_count > 0, (local[index, lower] + local[index, upper]) / 2.0, torch.ones_like(upper, dtype=local.dtype)))
    scale = median.clamp_min(1.0)
    score = torch.exp(-delta / (float(distance_temperature) * scale.unsqueeze(1)))
    return torch.where(effective_mask, float(prior_scale) * score, torch.zeros_like(score))


def _candidate_edge_index(graph: dict) -> torch.Tensor:
    """Build the undirected observable receiver/donor candidate graph."""
    donors, donor_mask, source = (graph['donors'], graph['candidate_mask'], graph['source'])
    edge_pairs = set()
    for slot in range(donors.shape[1]):
        m = donor_mask[:, slot]
        if torch.any(m):
            for s, d in zip(source[m].cpu().tolist(), donors[m, slot].cpu().tolist()):
                if s != d:
                    edge_pairs.add((s, d))
                    edge_pairs.add((d, s))
    edge_list = sorted(edge_pairs)
    if not edge_list:
        return torch.zeros((2, 0), dtype=torch.long, device=source.device)
    return torch.tensor(edge_list, dtype=torch.long, device=source.device).t()

class GATBaseline(nn.Module):
    """Graph Attention Network over the observable candidate graph."""

    def __init__(self) -> None:
        n_genes = 1000
        hidden = 512
        heads = 4
        n_layers = 3
        super().__init__()
        self.n_layers = n_layers
        convs, norms = ([], [])
        for i in range(n_layers):
            in_dim = n_genes if i == 0 else hidden * heads
            out_heads = 1 if i == n_layers - 1 else heads
            concat = i != n_layers - 1
            convs.append(pyg_nn.GATConv(in_dim, hidden, heads=out_heads, concat=concat))
            if concat:
                norms.append(nn.LayerNorm(hidden * heads))
            else:
                norms.append(nn.LayerNorm(hidden))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.head = nn.Linear(hidden, n_genes)

    def forward(self, graph: dict) -> torch.Tensor:
        edge_index = _candidate_edge_index(graph)
        if edge_index.numel() == 0:
            pred_clean = graph['initial'].clone()
        else:
            x = graph['initial']
            for conv, norm in zip(self.convs, self.norms):
                h = conv(x, edge_index)
                if h.shape[-1] == x.shape[-1]:
                    x = norm(torch.relu(h) + x)
                else:
                    x = norm(torch.relu(h))
            delta = self.head(x)
            pred_clean = graph['initial'] + delta
        return pred_clean

class BipartiteMPNNBaseline(nn.Module):
    """Bipartite MPNN with vectorized donor aggregation for memory efficiency."""

    def __init__(self) -> None:
        n_genes = 1000
        hidden = 256
        n_layers = 2
        super().__init__()
        self.n_layers = n_layers
        self.msg_net = nn.Sequential(nn.Linear(n_genes * 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_genes))
        self.gru = nn.GRUCell(n_genes, n_genes)
        self.bn = nn.BatchNorm1d(n_genes)

    def forward(self, graph: dict) -> torch.Tensor:
        mixed, source = (graph['mixed'], graph['source'])
        donors = graph['donors']
        fraction = fixed_distance_fraction(graph['distance'], graph['candidate_mask'])
        donor_mask = graph['candidate_mask']
        profile = graph['initial'].clone()
        for _ in range(self.n_layers):
            pred_per_obs = profile[source]
            slot_anchors = profile[donors.clamp_min(0)]
            slot_contrib = slot_anchors * fraction.unsqueeze(-1) * donor_mask.unsqueeze(-1).float()
            pred_per_obs = pred_per_obs + slot_contrib.sum(dim=1)
            residual = pred_per_obs - mixed
            msg_sum = torch.zeros_like(profile)
            msg_sum.index_add_(0, source, self.msg_net(torch.cat([residual, profile[source]], dim=-1)))
            donor_residual = residual.unsqueeze(1).expand(-1, donors.shape[1], -1)
            donor_profile = profile[donors.clamp_min(0)]
            donor_msgs = self.msg_net(torch.cat([donor_residual, donor_profile], dim=-1))
            donor_msgs = donor_msgs * donor_mask.unsqueeze(-1).float()
            safe_donors = donors.clamp_min(0)
            for slot in range(donors.shape[1]):
                msg_sum.index_add_(0, safe_donors[:, slot], donor_msgs[:, slot])
            msg_cnt = torch.zeros_like(profile)
            msg_cnt.index_add_(0, source, torch.ones_like(profile[source]))
            for slot in range(donors.shape[1]):
                msg_cnt.index_add_(0, safe_donors[:, slot], donor_mask[:, slot].unsqueeze(-1) * torch.ones_like(profile[source]))
            msg_sum = msg_sum / msg_cnt.clamp_min(1.0)
            delta = self.gru(msg_sum, profile)
            delta = self.bn(delta)
            profile = delta * 1.0 + profile
        return profile

class _GraphConvBaselineBase(nn.Module):
    """Shared graph-convolution stack with a residual expression output."""

    def __init__(self, n_genes, hidden, n_layers) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.convs = self._build_convs(n_genes, hidden, n_layers)
        self.norms = nn.ModuleList((nn.LayerNorm(dim) for dim in self._norm_dims(n_genes, hidden, n_layers)))
        self.head = nn.Linear(hidden, n_genes)

    def _build_convs(self, n_genes: int, hidden: int, n_layers: int) -> nn.ModuleList:
        raise NotImplementedError

    def _norm_dims(self, n_genes: int, hidden: int, n_layers: int) -> list[int]:
        return [hidden] * n_layers

    def forward(self, graph: dict) -> torch.Tensor:
        edge_index = _candidate_edge_index(graph)
        if edge_index.numel() == 0:
            pred_clean = graph['initial'].clone()
        else:
            x = graph['initial']
            for conv, norm in zip(self.convs, self.norms):
                h = conv(x, edge_index)
                if h.shape[-1] == x.shape[-1]:
                    x = norm(torch.relu(h) + x)
                else:
                    x = norm(torch.relu(h))
            delta = self.head(x)
            pred_clean = graph['initial'] + delta
        return pred_clean

class GCNBaseline(_GraphConvBaselineBase):
    """Graph Convolutional Network baseline."""

    def __init__(self):
        super().__init__(1000, 512, 3)

    def _build_convs(self, n_genes: int, hidden: int, n_layers: int) -> nn.ModuleList:
        in_dims = [n_genes] + [hidden] * (n_layers - 1)
        return nn.ModuleList((pyg_nn.GCNConv(d, hidden) for d in in_dims))

class GraphSAGEBaseline(_GraphConvBaselineBase):
    """GraphSAGE baseline."""

    def __init__(self):
        super().__init__(1000, 512, 2)

    def _build_convs(self, n_genes: int, hidden: int, n_layers: int) -> nn.ModuleList:
        in_dims = [n_genes] + [hidden] * (n_layers - 1)
        return nn.ModuleList((pyg_nn.SAGEConv(d, hidden) for d in in_dims))

class GATv2Baseline(_GraphConvBaselineBase):
    """Graph Attention Network v2 baseline (mirrors GATBaseline layer shapes)."""

    def __init__(self) -> None:
        n_genes = 1000
        hidden = 256
        heads = 4
        n_layers = 2
        dropout = 0.0
        self.heads = int(heads)
        self.dropout = float(dropout)
        super().__init__(n_genes, hidden, n_layers)

    def _build_convs(self, n_genes: int, hidden: int, n_layers: int) -> nn.ModuleList:
        convs = []
        for i in range(n_layers):
            in_dim = n_genes if i == 0 else hidden * self.heads
            out_heads = 1 if i == n_layers - 1 else self.heads
            concat = i != n_layers - 1
            convs.append(pyg_nn.GATv2Conv(in_dim, hidden, heads=out_heads, concat=concat, dropout=self.dropout))
        return nn.ModuleList(convs)

    def _norm_dims(self, n_genes: int, hidden: int, n_layers: int) -> list[int]:
        return [hidden * self.heads if i != n_layers - 1 else hidden for i in range(n_layers)]
