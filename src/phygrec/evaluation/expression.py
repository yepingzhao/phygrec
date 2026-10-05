"""Receiver-scene expression metrics for validation and held-out test data."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phygrec.data.hashing import sha256_file
from phygrec.data.shared_scene_graphs import SharedSceneGraphDataset, pack_scene_graphs
from phygrec.models.inputs import build_compliant_model_input
from phygrec.protocol import DATA, FOLDS, ROOT, SEEDS, SPLITS, benchmark_directory
from phygrec.evaluation.expression_metrics import ExpressionMetricAccumulator


@torch.inference_mode()
def evaluate_validation(module, store_root: Path, *, device: torch.device,
                 workers: int) -> dict:
    store = store_root / "val.h5"
    dataset = SharedSceneGraphDataset(store)
    loader = DataLoader(dataset, batch_size=6, shuffle=False,
                        collate_fn=pack_scene_graphs, num_workers=workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0)
    overall = ExpressionMetricAccumulator()
    by_combination = defaultdict(ExpressionMetricAccumulator)
    for cpu in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in cpu.items()}
        source = batch["source"]
        prediction = module(build_compliant_model_input(batch))
        initial = batch["initial"][source]
        pred = prediction[source]
        clean = batch["clean"][source]
        overall.update(initial, pred, clean)
        graph_index = batch["observation_graph_index"]
        for index, name in enumerate(batch["combination"]):
            keep = graph_index == index
            by_combination[name].update(initial[keep], pred[keep], clean[keep])
    return {
        "store": os.path.relpath(store, ROOT),
        "store_sha256": sha256_file(store),
        "aggregation": "receiver_cell_scene_entry_micro",
        "overall": overall.summary(),
        "by_combination": {name: row.summary() for name, row in sorted(by_combination.items())},
    }


def evaluate_model(model, split: str, seed: int, device_name: str, *,
                   structure: bool = False, batch_size: int = 6, baseline: bool = False) -> dict:
    device = torch.device(device_name)
    dataset = SharedSceneGraphDataset(benchmark_directory(split) / "test.h5")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=pack_scene_graphs, num_workers=0)
    overall = ExpressionMetricAccumulator()
    labels_seen: set[str] = set()
    scenes_seen: set[str] = set()
    cells = None
    if structure:
        from phygrec.evaluation.structure import PhysicalCellAccumulator
        cells = PhysicalCellAccumulator(split, dataset.genes)
    with torch.inference_mode():
        for cpu in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in cpu.items()}
            prediction = model(batch if baseline else build_compliant_model_input(batch))
            source = batch["source"]
            labels = np.asarray(batch["labels"])[source.cpu().numpy()].astype(str)
            scene_ids = np.asarray(batch["scene_id"])[batch["observation_graph_index"].cpu().numpy()]
            if split == "main":
                selected_np = np.ones(len(source), dtype=bool)
            else:
                heldout, primary = FOLDS[split]
                combinations = np.asarray(batch["combination"])[batch["observation_graph_index"].cpu().numpy()]
                chips = np.asarray([label.rsplit("_", 1)[0] for label in labels])
                selected_np = np.isin(combinations, primary) & (chips == heldout)
            if not selected_np.any():
                continue
            nodes = source[torch.as_tensor(selected_np, device=device)]
            overall.update(batch["initial"][nodes], prediction[nodes], batch["clean"][nodes])
            if cells is not None:
                cells.update(labels[selected_np], batch["clean"][nodes].cpu().numpy(),
                             batch["initial"][nodes].cpu().numpy(), prediction[nodes].cpu().numpy())
            labels_seen.update(labels[selected_np].tolist())
            scenes_seen.update(scene_ids[selected_np].tolist())
    if not overall.entries:
        raise ValueError("No test receiver-scene instances were selected")
    scores = overall.summary()
    result = {
        "split": split, "seed": seed, "receiver_scene_entries": overall.entries,
        "unique_physical_receivers": len(labels_seen), "scenes": len(scenes_seen),
        "genes": overall.genes,
        "count_mae": scores["pred_raw_mae"],
        "count_rmse": scores["count_rmse"],
        "log_mae": scores["pred_profile_mae"],
        "log_rmse": scores["log_rmse"],
        "relative_l1": scores["pred_raw_l1"],
        "relative_l2": scores["relative_l2"],
        "input_count_mae": scores["input_raw_mae"],
        "input_log_mae": scores["input_profile_mae"],
        "input_relative_l1": scores["input_raw_l1"],
    }
    if cells is not None:
        from phygrec.evaluation.structure import score_cells
        result["structure"] = score_cells(cells.means(), split)
    return result


def evaluate(split: str, seed: int, device_name: str,
             checkpoint_path: Path | None = None, *, structure: bool = False) -> dict:
    from phygrec.training.module import PhyGRecModule

    checkpoint = checkpoint_path or DATA / f"checkpoints/{split}/seed{seed}.ckpt"
    model = PhyGRecModule.load_from_checkpoint(str(checkpoint), map_location="cpu").to(device_name).eval()
    return evaluate_model(model, split, seed, device_name, structure=structure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint from a newly trained run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--structure", action="store_true", help="Also score physical-cell annotation and clustering")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.split, args.seed, args.device, args.checkpoint, structure=args.structure), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
