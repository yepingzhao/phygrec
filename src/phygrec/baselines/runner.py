"""Train and evaluate the seven fixed baselines on main and receiver-LOCO splits."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from phygrec.baselines.gnn import (
    GATBaseline, GATv2Baseline, GCNBaseline, GraphSAGEBaseline, BipartiteMPNNBaseline,
)
from phygrec.baselines.physical import FittedCircleOperator, solve_physical_pgd
from phygrec.baselines.vae import VAEBaseline
from phygrec.data.shared_scene_graphs import SharedSceneGraphDataset, pack_scene_graphs
from phygrec.expression_metrics import ExpressionMetricAccumulator
from phygrec.evaluation import evaluate_model
from phygrec.operators import graph_prediction
from phygrec.protocol import ROOT, benchmark_directory
from phygrec.transforms import normalized_log


MODELS = {
    "gatv2": GATv2Baseline, "graphsage": GraphSAGEBaseline,
    "gat": GATBaseline, "gcn": GCNBaseline, "mpnn": BipartiteMPNNBaseline,
    "vae": VAEBaseline,
}
METHODS = (*MODELS, "physical_pgd")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown baseline: {method}")
    config = yaml.safe_load((ROOT / f"configs/baselines/{method}.yaml").read_text())
    architecture = {
        "gatv2": {"hidden": 256, "n_layers": 2, "heads": 4},
        "graphsage": {"hidden": 512, "n_layers": 2},
        "gat": {"hidden": 512, "n_layers": 3, "heads": 4},
        "gcn": {"hidden": 512, "n_layers": 3},
        "mpnn": {"hidden": 256, "n_layers": 2},
        "vae": {"hidden": 512, "latent": 32},
        "physical_pgd": {"maximum_row_crosstalk": 0.8},
    }[method]
    if any(config[key] != value for key, value in architecture.items()):
        raise ValueError("Baseline architecture must match the fixed published model")
    return config


class BaselinePredictor(torch.nn.Module):
    """Keep targets outside every baseline's inference inputs."""

    def __init__(self, method: str) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown baseline: {method}")
        self.method = method
        self.model = FittedCircleOperator() if method == "physical_pgd" else MODELS[method]()

    def forward(self, batch: dict) -> torch.Tensor:
        if self.method == "physical_pgd":
            total = batch["initial"].sum(dim=1)
            fraction, _ = self.model(
                batch["distance"], batch["candidate_mask"], total[batch["source"]],
                total[batch["donors"].clamp_min(0)], batch.get("observation_graph_index"),
            )
            config = load_config(self.method)
            return solve_physical_pgd(batch, fraction, config["step_size"], config["iterations"])[0]
        keys = ("mixed", "source", "initial") if self.method == "vae" else (
            "mixed", "source", "initial", "donors", "distance", "candidate_mask",
        )
        prediction, _ = self.model({key: batch[key] for key in keys})
        return prediction.clamp_min(0.0)


def baseline_loss(model: BaselinePredictor, prediction: torch.Tensor, batch: dict) -> torch.Tensor:
    mask = batch["train_mask"]
    clean, predicted = batch["clean"][mask], prediction[mask]
    raw = ((predicted - clean).abs().sum(dim=1) / clean.sum(dim=1).clamp_min(1.0)).mean()
    profile = (normalized_log(predicted) - normalized_log(clean)).abs().mean()
    loss = raw + profile
    if model.method == "vae":
        loss = loss + float(load_config("vae")["kl_weight"]) * model.model._kl
    return loss


def make_scene_loader(path: Path, batch_size: int, shuffle: bool, workers: int, seed: int) -> DataLoader:
    return DataLoader(
        SharedSceneGraphDataset(path), batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), collate_fn=pack_scene_graphs,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None, drop_last=False,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


@torch.inference_mode()
def evaluate_validation(model: BaselinePredictor, data: DataLoader, device: torch.device) -> dict:
    model.eval()
    totals = ExpressionMetricAccumulator()
    for cpu in data:
        batch = move_batch_to_device(cpu, device)
        source = batch["source"]
        totals.update(batch["initial"][source], model(batch)[source], batch["clean"][source])
    return totals.summary()


def physical_fit_loss(operator: FittedCircleOperator, batch: dict) -> torch.Tensor:
    """Fit geometry from observable totals and train-only labeled profiles."""
    total = batch["initial"].sum(dim=1)
    fraction, _ = operator(
        batch["distance"], batch["candidate_mask"], total[batch["source"]],
        total[batch["donors"].clamp_min(0)], batch.get("observation_graph_index"),
    )
    predicted = graph_prediction(
        batch["clean"], batch["source"], batch["donors"], fraction, batch["candidate_mask"],
    )
    observed = batch["mixed"]
    nll = (predicted.clamp_min(1e-8) - observed * torch.log(predicted.clamp_min(1e-8))
           + torch.lgamma(observed + 1.0))
    return nll.mean() / observed.mean().clamp_min(1.0)


def fit_circle_operator(model: BaselinePredictor, split: str, seed: int,
                 device: torch.device, workers: int) -> None:
    config = load_config("physical_pgd")
    dataset = SharedSceneGraphDataset(benchmark_directory(split) / "train.h5")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["fit_lr"],
                                  weight_decay=config["fit_weight_decay"])
    if split == "main":
        # Main fitting averages three independent scene losses before each update.
        data = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=list,
                          num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                          prefetch_factor=2 if workers > 0 else None)
    for epoch in range(config["fit_epochs"]):
        if split != "main":
            # The original LOCO fit packs evenly distributed six-scene batches.
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(
                seed + 1_000_003 * epoch)).tolist()
            count = (len(dataset) + 5) // 6
            base, extra = divmod(len(dataset), count)
            sizes = [base + int(i < extra) for i in range(count)]
            offsets = np.cumsum([0, *sizes])
            batches = [indices[offsets[i]:offsets[i + 1]] for i in range(count)]
            data = DataLoader(dataset, batch_sampler=batches, collate_fn=pack_scene_graphs,
                              num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                              prefetch_factor=2 if workers > 0 else None)
        for cpu in data:
            optimizer.zero_grad(set_to_none=True)
            scenes = cpu if split == "main" else [cpu]
            for scene in scenes:
                loss = physical_fit_loss(model.model, move_batch_to_device(scene, device))
                (loss / len(scenes)).backward()
            optimizer.step()
        print(json.dumps({"fit_epoch": epoch + 1}), flush=True)


def train(method: str, split: str, seed: int, device_name: str, workers: int) -> Path:
    config = load_config(method)
    run = ROOT / f"runs/baselines/{method}/{split}/seed{seed}"
    run.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    device = torch.device(device_name)
    model = BaselinePredictor(method).to(device)
    checkpoint = run / "best.pt"

    def save(epoch: int, score: dict | None) -> None:
        torch.save({"method": method, "split": split, "seed": seed, "config": config,
                    "epoch": epoch, "selection_split": "train_only" if score is None else "val_only",
                    "val": score, "state_dict": model.state_dict()}, checkpoint)

    if method == "physical_pgd":
        fit_circle_operator(model, split, seed, device, workers)
        save(config["fit_epochs"], None)
        return checkpoint
    directory = benchmark_directory(split)
    train_data = make_scene_loader(directory / "train.h5", config["batch_size"], True, workers, seed)
    val_data = make_scene_loader(directory / "val.h5", config["batch_size"], False, workers, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best_score = -float("inf")
    with (run / "history.jsonl").open("x") as history:
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            for cpu in train_data:
                batch = move_batch_to_device(cpu, device)
                optimizer.zero_grad(set_to_none=True)
                loss = baseline_loss(model, model(batch), batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
            if epoch == 1 or epoch % config["eval_every"] == 0:
                score = evaluate_validation(model, val_data, device)
                value = score["combined_recovery"]
                if not np.isfinite(value):
                    raise RuntimeError(f"Non-finite validation score at epoch {epoch}")
                if value > best_score:
                    best_score = value
                    save(epoch, score)
                row = {"epoch": epoch, "val": score}
                history.write(json.dumps(row, allow_nan=False) + "\n")
                history.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
    return checkpoint


def evaluate(method: str, split: str, seed: int, checkpoint: Path, device_name: str,
             *, structure: bool = False) -> dict:
    config = load_config(method)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (raw["method"], raw["split"], raw["seed"], raw["config"]) != (method, split, seed, config):
        raise ValueError("Checkpoint does not match the fixed baseline run")
    expected = "train_only" if method == "physical_pgd" else "val_only"
    if raw["selection_split"] != expected:
        raise ValueError("Checkpoint has an invalid selection split")
    seed_everything(seed)
    model = BaselinePredictor(method).to(device_name).eval()
    model.load_state_dict(raw["state_dict"], strict=True)
    batch_size = 6 if split != "main" or method == "physical_pgd" else config["batch_size"]
    result = evaluate_model(model, split, seed, device_name, structure=structure,
                            batch_size=batch_size, baseline=True)
    return {"method": method, **result, "selected_epoch": raw["epoch"]}
