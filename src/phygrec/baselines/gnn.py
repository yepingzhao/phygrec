"""The five fixed graph baselines used in the paper."""

from __future__ import annotations
import torch
from torch import nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from phygrec.baselines.physical import graph_fraction

def _build_source_source_graph(graph: dict, node_features: torch.Tensor | None=None) -> Data:
    """Build graph with source-source and donor-source edges."""
    donors, donor_mask, source = (graph['donors'], graph['candidate_mask'], graph['source'])
    edge_pairs = set()
    for slot in range(donors.shape[1]):
        m = donor_mask[:, slot]
        if torch.any(m):
            for s, d in zip(source[m].cpu().tolist(), donors[m, slot].cpu().tolist()):
                if s != d:
                    edge_pairs.add((min(s, d), max(s, d)))
                    edge_pairs.add((s, d))
                    edge_pairs.add((d, s))
    edge_list = sorted(edge_pairs)
    if not edge_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=source.device)
    else:
        edges = torch.tensor(edge_list, dtype=torch.long, device=source.device).t()
        edge_index = edges
    return Data(x=graph['initial'] if node_features is None else node_features, edge_index=edge_index)

class GATBaseline(nn.Module):
    """Graph Attention Network over source-source + donor-source graph."""

    def __init__(self) -> None:
        n_genes = 1000
        hidden = 512
        heads = 4
        n_layers = 3
        prior_scale = 0.5
        distance_temperature = 1.0
        super().__init__()
        self.n_layers = n_layers
        self.prior_scale = float(prior_scale)
        self.distance_temperature = float(distance_temperature)
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

    def forward(self, graph: dict) -> tuple[torch.Tensor, torch.Tensor]:
        data = _build_source_source_graph(graph)
        if data.edge_index.numel() == 0:
            pred_clean = graph['initial'].clone()
        else:
            x = data.x
            for conv, norm in zip(self.convs, self.norms):
                h = conv(x, data.edge_index)
                if h.shape[-1] == x.shape[-1]:
                    x = norm(torch.relu(h) + x)
                else:
                    x = norm(torch.relu(h))
            delta = self.head(x)
            pred_clean = graph['initial'] + delta
        mixed, source = (graph['mixed'], graph['source'])
        donors = graph['donors']
        fraction = graph_fraction(graph, self.prior_scale, self.distance_temperature)
        donor_mask = graph['candidate_mask']
        safe_donors = donors.clamp_min(0)
        pred_obs = pred_clean[source] + (pred_clean[safe_donors] * fraction.unsqueeze(-1) * donor_mask.unsqueeze(-1).float()).sum(dim=1)
        residual = pred_obs - mixed
        return (pred_clean, residual)

class BipartiteMPNNBaseline(nn.Module):
    """Bipartite MPNN with vectorized donor aggregation for memory efficiency."""

    def __init__(self) -> None:
        n_genes = 1000
        hidden = 256
        n_layers = 2
        prior_scale = 0.5
        distance_temperature = 1.0
        super().__init__()
        self.n_layers = n_layers
        self.prior_scale = float(prior_scale)
        self.distance_temperature = float(distance_temperature)
        self.msg_net = nn.Sequential(nn.Linear(n_genes * 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_genes))
        self.gru = nn.GRUCell(n_genes, n_genes)
        self.bn = nn.BatchNorm1d(n_genes)

    def forward(self, graph: dict) -> tuple[torch.Tensor, torch.Tensor]:
        mixed, source = (graph['mixed'], graph['source'])
        donors = graph['donors']
        fraction = graph_fraction(graph, self.prior_scale, self.distance_temperature)
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
        pred_per_obs = profile[source]
        slot_anchors = profile[donors.clamp_min(0)]
        slot_contrib = slot_anchors * fraction.unsqueeze(-1) * donor_mask.unsqueeze(-1).float()
        final_residual = pred_per_obs + slot_contrib.sum(dim=1) - mixed
        return (profile, final_residual)

class _GraphConvBaselineBase(nn.Module):
    """Shared graph-conv baseline: conv stack over the source-source graph,
    residual delta output, then physical residual via graph_fraction."""

    def __init__(self, n_genes, hidden, n_layers) -> None:
        prior_scale = 0.5
        distance_temperature = 1.0
        super().__init__()
        self.n_layers = n_layers
        self.prior_scale = float(prior_scale)
        self.distance_temperature = float(distance_temperature)
        self.convs = self._build_convs(n_genes, hidden, n_layers)
        self.norms = nn.ModuleList((nn.LayerNorm(dim) for dim in self._norm_dims(n_genes, hidden, n_layers)))
        self.head = nn.Linear(self._head_in_dim(hidden), n_genes)

    def _build_convs(self, n_genes: int, hidden: int, n_layers: int) -> nn.ModuleList:
        raise NotImplementedError

    def _norm_dims(self, n_genes: int, hidden: int, n_layers: int) -> list[int]:
        return [hidden] * n_layers

    def _head_in_dim(self, hidden: int) -> int:
        return hidden

    def forward(self, graph: dict) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = graph['initial']
        data = _build_source_source_graph(graph, node_features)
        if data.edge_index.numel() == 0:
            pred_clean = graph['initial'].clone()
        else:
            x = data.x
            for conv, norm in zip(self.convs, self.norms):
                h = conv(x, data.edge_index)
                if h.shape[-1] == x.shape[-1]:
                    x = norm(torch.relu(h) + x)
                else:
                    x = norm(torch.relu(h))
            delta = self.head(x)
            pred_clean = graph['initial'] + delta
        mixed, source = (graph['mixed'], graph['source'])
        donors = graph['donors']
        fraction = graph_fraction(graph, self.prior_scale, self.distance_temperature)
        donor_mask = graph['candidate_mask']
        safe_donors = donors.clamp_min(0)
        pred_obs = pred_clean[source] + (pred_clean[safe_donors] * fraction.unsqueeze(-1) * donor_mask.unsqueeze(-1).float()).sum(dim=1)
        residual = pred_obs - mixed
        return (pred_clean, residual)

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
        prior_scale = 0.5
        distance_temperature = 1.0
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
