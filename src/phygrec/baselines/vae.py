"""The fixed supervised VAE comparison model."""

import torch
from torch import nn


class VAEBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1000, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
        )
        self.mu, self.logvar = nn.Linear(512, 32), nn.Linear(512, 32)
        self.decoder = nn.Sequential(
            nn.Linear(32, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
        )
        self.mean_head = nn.Linear(512, 1000)

    def forward(self, graph: dict) -> tuple[torch.Tensor, torch.Tensor]:
        mixed, source = graph["mixed"], graph["source"]
        h = self.encoder(mixed)
        mu, logvar = self.mu(h), self.logvar(h).clamp(min=-10.0, max=10.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) if self.training else mu
        self._kl = (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)).sum(dim=1).mean()
        pred_per_obs = self.mean_head(self.decoder(z))
        pred_clean = torch.zeros(len(graph["initial"]), mixed.shape[1], device=mixed.device)
        count = torch.zeros(len(graph["initial"]), device=mixed.device)
        pred_clean.index_add_(0, source, pred_per_obs)
        count.index_add_(0, source, torch.ones(len(source), device=mixed.device))
        pred_clean = pred_clean / count.clamp_min(1.0).unsqueeze(1)
        return pred_clean + graph["initial"], torch.zeros_like(mixed)
