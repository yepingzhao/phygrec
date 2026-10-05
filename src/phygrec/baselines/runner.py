"""Train and evaluate the seven fixed baselines on main and receiver-LOCO splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from phygrec.baselines.gnn import (
    GATBaseline, GATv2Baseline, GCNBaseline, GraphSAGEBaseline, BipartiteMPNNBaseline,
)
from phygrec.baselines.physical import FittedCircleOperator, physical_solve
from phygrec.baselines.vae import VAEBaseline
from phygrec.data.shared_scene_graphs import SharedSceneGraphDataset, pack_scene_graphs
from phygrec.evaluation import Totals, evaluate_model
from phygrec.operators import graph_prediction
from phygrec.protocol import ROOT, SEEDS, SPLITS, benchmark_directory
from phygrec.reproduction import summarize_results
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


class BaselinePrediction(torch.nn.Module):
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
            return physical_solve(batch, fraction, config["step_size"], config["iterations"])[0]
        keys = ("mixed", "source", "initial") if self.method == "vae" else (
            "mixed", "source", "initial", "donors", "distance", "candidate_mask",
        )
        prediction, _ = self.model({key: batch[key] for key in keys})
        return prediction.clamp_min(0.0)


def objective(model: BaselinePrediction, prediction: torch.Tensor, batch: dict) -> torch.Tensor:
    mask = batch["train_mask"]
    clean, predicted = batch["clean"][mask], prediction[mask]
    raw = ((predicted - clean).abs().sum(dim=1) / clean.sum(dim=1).clamp_min(1.0)).mean()
    profile = (normalized_log(predicted) - normalized_log(clean)).abs().mean()
    loss = raw + profile
    if model.method == "vae":
        loss = loss + float(load_config("vae")["kl_weight"]) * model.model._kl
    return loss


def loader(path: Path, batch_size: int, shuffle: bool, workers: int, seed: int) -> DataLoader:
    return DataLoader(
        SharedSceneGraphDataset(path), batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), collate_fn=pack_scene_graphs,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None, drop_last=False,
    )


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


@torch.inference_mode()
def validation_score(model: BaselinePrediction, data: DataLoader, device: torch.device) -> dict:
    model.eval()
    totals = Totals()
    for cpu in data:
        batch = move(cpu, device)
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


def fit_physical(model: BaselinePrediction, split: str, seed: int,
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
                loss = physical_fit_loss(model.model, move(scene, device))
                (loss / len(scenes)).backward()
            optimizer.step()
        print(json.dumps({"fit_epoch": epoch + 1}), flush=True)


def train(method: str, split: str, seed: int, device_name: str, workers: int) -> Path:
    config = load_config(method)
    run = ROOT / f"runs/baselines/{method}/{split}/seed{seed}"
    run.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    device = torch.device(device_name)
    model = BaselinePrediction(method).to(device)
    checkpoint = run / "best.pt"

    def save(epoch: int, score: dict | None) -> None:
        torch.save({"method": method, "split": split, "seed": seed, "config": config,
                    "epoch": epoch, "selection_split": "train_only" if score is None else "val_only",
                    "val": score, "state_dict": model.state_dict()}, checkpoint)

    if method == "physical_pgd":
        fit_physical(model, split, seed, device, workers)
        save(config["fit_epochs"], None)
        return checkpoint
    directory = benchmark_directory(split)
    train_data = loader(directory / "train.h5", config["batch_size"], True, workers, seed)
    val_data = loader(directory / "val.h5", config["batch_size"], False, workers, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best_score = -float("inf")
    with (run / "history.jsonl").open("x") as history:
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            for cpu in train_data:
                batch = move(cpu, device)
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, model(batch), batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
            if epoch == 1 or epoch % config["eval_every"] == 0:
                score = validation_score(model, val_data, device)
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
    model = BaselinePrediction(method).to(device_name).eval()
    model.load_state_dict(raw["state_dict"], strict=True)
    batch_size = 6 if split != "main" or method == "physical_pgd" else config["batch_size"]
    result = evaluate_model(model, split, seed, device_name, structure=structure,
                            batch_size=batch_size, baseline=True)
    return {"method": method, **result, "selected_epoch": raw["epoch"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "reproduce"))
    parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    parser.add_argument("--split", choices=SPLITS, default="main")
    parser.add_argument("--experiment", choices=("main", "loco", "all"), default="main")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--seeds", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action != "reproduce":
        if args.method == "all":
            parser.error("Individual actions require one method")
        if args.action == "train":
            print(train(args.method, args.split, args.seed, args.device, args.workers))
        else:
            checkpoint = args.checkpoint or ROOT / f"runs/baselines/{args.method}/{args.split}/seed{args.seed}/best.pt"
            print(json.dumps(evaluate(args.method, args.split, args.seed, checkpoint,
                                      args.device, structure=args.structure), indent=2, allow_nan=False))
        return
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Each seed must occur once")
    methods = METHODS if args.method == "all" else (args.method,)
    splits = ("main",) if args.experiment == "main" else SPLITS[1:] if args.experiment == "loco" else SPLITS
    plan = []
    for method in methods:
        for split in splits:
            for seed in args.seeds:
                common = ["--method", method, "--split", split, "--seed", str(seed),
                          "--device", args.device, "--workers", str(args.workers)]
                commands = []
                if not args.evaluate_only:
                    commands.append([sys.executable, "scripts/baselines.py", "train", *common])
                commands.append([sys.executable, "scripts/baselines.py", "evaluate", *common])
                if args.structure:
                    commands[-1].append("--structure")
                plan.append({"method": method, "split": split, "seed": seed, "commands": commands})
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    rows = {method: [] for method in methods}
    output = ROOT / "runs/results/baselines"
    for item in plan:
        for command in item["commands"][:-1]:
            subprocess.run(command, cwd=ROOT, check=True)
        # Scoring is also isolated so reconstruction of each run has the original RNG state.
        response = subprocess.run(item["commands"][-1], cwd=ROOT, check=True, capture_output=True, text=True)
        result = json.loads(response.stdout)
        directory = output / item["method"] / item["split"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"seed{item['seed']}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        rows[item["method"]].append(result)
    summary = {method: summarize_results(results) for method, results in rows.items()}
    (output / f"{args.experiment}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
