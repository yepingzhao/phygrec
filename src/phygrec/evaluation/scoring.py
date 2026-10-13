"""Aggregate physical cells, run scoring and summarize folds and training seeds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from phygrec import DATA, SPLITS, VARIANTS
from phygrec.utils import package_versions, sha256_file, single_threaded_scoring


RESOURCES = DATA / "structure"


EXPRESSION_METRICS = ("count_mae", "log_mae", "relative_l1", "count_rmse", "log_rmse", "relative_l2")


STRUCTURE_METRICS = (
    "accuracy", "macro_precision", "balanced_accuracy", "macro_f1", "mcc",
    "cohen_kappa", "macro_average_precision", "source_agreement_ari",
    "source_agreement_nmi", "leiden_stability_ari", "frozen_label_silhouette_20pc",
    "leiden_clusters",
)


def load_reference(split: str, genes: np.ndarray) -> tuple[dict, dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    metadata = json.loads((RESOURCES / "manifest.json").read_text())[split]
    path = RESOURCES / f"{split}.npz"
    if sha256_file(path) != metadata["labels_sha256"]:
        raise ValueError("Frozen structure labels changed")
    with np.load(path, allow_pickle=False) as bank:
        reference = {name: bank[name] for name in ("ids", "genes", "clusters")}
    if not np.array_equal(genes, reference["genes"]):
        raise ValueError("Structure gene panel differs from frozen Reference")
    return reference, metadata


class PhysicalCellAccumulator:
    """Accumulate raw float64 sums, then cast identity means to float32."""

    def __init__(self, split: str, genes: np.ndarray) -> None:
        self.reference, _ = load_reference(split, genes)
        self.index = {label: index for index, label in enumerate(self.reference["ids"])}
        self.counts = np.zeros(len(self.index), dtype=np.int64)
        shape = (len(self.index), len(genes))
        self.sums = {key: np.zeros(shape, dtype=np.float64)
                     for key in ("clean", "mixed", "prediction")}

    def update(self, labels: np.ndarray, clean: np.ndarray,
               mixed: np.ndarray, prediction: np.ndarray) -> None:
        positions = np.asarray([self.index.get(label, -1) for label in labels])
        keep = positions >= 0
        if np.any(clean[~keep].sum(axis=1, dtype=np.float64) > 0):
            raise ValueError("Nonzero receiver is absent from frozen structure cohort")
        np.add.at(self.counts, positions[keep], 1)
        for key, values in (("clean", clean), ("mixed", mixed), ("prediction", prediction)):
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite structure expression: {key}")
            np.add.at(self.sums[key], positions[keep], values[keep])

    def means(self) -> dict:
        if np.any(self.counts == 0):
            raise ValueError("Missing receiver identities from frozen structure cohort")
        return {"ids": self.reference["ids"], "genes": self.reference["genes"],
                **{key: (values / self.counts[:, None]).astype(np.float32)
                   for key, values in self.sums.items()}}


def score_cells(cells: dict, split: str) -> dict:
    """Run scoring in a separate process with the paper's thread settings."""
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    with tempfile.TemporaryDirectory(prefix="phygrec-structure-") as directory:
        path = Path(directory) / "cells.npz"
        np.savez_compressed(path, **cells)
        completed = subprocess.run(
            [sys.executable, "-m", "phygrec.evaluation.scoring", "--split", split, "--cells", str(path)],
            env=environment, capture_output=True, text=True, check=True,
        )
    return json.loads(completed.stdout)


def evaluate_cells(cells: dict, split: str) -> dict:
    """Score only the frozen nonzero Reference cohort, after prediction."""
    from phygrec.evaluation.annotation import (
        CLASS_ORDER, annotation_metric_record, annotation_protocol,
    )
    from phygrec.evaluation.clustering import clustering_metrics, independent_pca

    with single_threaded_scoring():
        reference, metadata = load_reference(split, cells["genes"])
        if not np.array_equal(cells["ids"], reference["ids"]):
            raise ValueError("Structure identities/order differ from frozen Reference")
        clean = np.ascontiguousarray(cells["clean"], dtype=np.float32)
        if hashlib.sha256(clean.tobytes()).hexdigest() != metadata["clean_sha256"]:
            raise ValueError("Structure Reference expression changed")
        scaler, labels, encoded, marker_indices, markers = annotation_protocol(clean, cells["genes"])
        scores = {}
        for state, key in (("reference", "clean"), ("mixed", "mixed"), ("prediction", "prediction")):
            matrix = np.asarray(cells[key], dtype=np.float32)
            if matrix.shape != clean.shape or not np.isfinite(matrix).all() or matrix.min() < -1e-5:
                raise ValueError(f"Invalid structure expression: {state}")
            matrix = np.maximum(matrix, 0)
            if np.any(matrix.sum(axis=1, dtype=np.float64) <= 0):
                raise ValueError(f"Zero-library structure expression: {state}")
            record = annotation_metric_record(matrix, scaler=scaler, reference_labels=labels,
                                             encoded_reference=encoded, marker_indices=marker_indices)
            if state == "reference":
                record["macro_average_precision"] = None
            scores[state] = {**record, **clustering_metrics(independent_pca(matrix), reference["clusters"])}
        return {"physical_cells": len(clean), "genes": clean.shape[1],
                "aggregation": "equal_raw_mean_by_physical_identity_per_training_seed",
                "class_order": list(CLASS_ORDER), "marker_genes_used": markers,
                "reference_labels_sha256": metadata["labels_sha256"],
                "reference_expression_sha256": metadata["clean_sha256"],
                "prediction_expression_sha256": hashlib.sha256(
                    np.ascontiguousarray(cells["prediction"], dtype=np.float32).tobytes()).hexdigest(),
                "evaluation": {
                    "numerical_threads": 1,
                    "packages": package_versions("numpy", "scipy", "scikit-learn", "scanpy", "anndata",
                                                 "numba", "llvmlite", "pynndescent", "umap-learn",
                                                 "igraph", "leidenalg", "threadpoolctl"),
                }, "states": scores}


def summarize_results(results: list[dict]) -> dict:
    """Use sample SD across seeds and equal fold weights within each LOCO seed."""
    groups = {}
    has_structure = ["structure" in row for row in results]
    if any(has_structure) and not all(has_structure):
        raise ValueError("Structure results must cover every run in the summary")
    loco = {}
    identities = set()
    for result in results:
        identity = (result["split"], result["seed"], result.get("variant"))
        split, _, variant = identity
        if split not in SPLITS or (variant is not None and
                                   (split != "main" or variant not in VARIANTS)):
            raise ValueError(f"Unpublished result identity: {identity}")
        if identity in identities:
            raise ValueError(f"Duplicate result: {identity}")
        identities.add(identity)
        if result["split"] != "main":
            loco.setdefault(result["seed"], {})[result["split"]] = result
        else:
            group = result.get("variant") or "main"
            groups.setdefault(group, []).append(result)
    if loco:
        groups["loco"] = []
        for seed, folds in sorted(loco.items()):
            if set(folds) != {"a9", "l7", "na"}:
                raise ValueError(f"LOCO seed {seed} requires all three folds")
            groups["loco"].append({
                "seed": seed,
                **{metric: float(np.mean([folds[fold][metric] for fold in ("a9", "l7", "na")]))
                   for metric in EXPRESSION_METRICS},
            })
            if all(has_structure):
                groups["loco"][-1]["structure"] = {"states": {
                    state: {metric: float(np.mean([folds[fold]["structure"]["states"][state][metric]
                                                   for fold in ("a9", "l7", "na")]))
                            for metric in STRUCTURE_METRICS
                            if not (state == "reference" and metric == "macro_average_precision")}
                    for state in ("reference", "mixed", "prediction")}}
    summary = {}
    for group, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["seed"])
        summary[group] = {
            "seeds": [row["seed"] for row in rows],
            "n_seeds": len(rows),
            "metrics": {
                metric: {
                    "mean": float(np.mean([row[metric] for row in rows])),
                    "sample_sd": float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else None,
                }
                for metric in EXPRESSION_METRICS
            },
        }
        if all(has_structure):
            summary[group]["structure"] = {
                state: {metric: {
                    "mean": float(np.mean([row["structure"]["states"][state][metric] for row in rows])),
                    "sample_sd": (float(np.std([row["structure"]["states"][state][metric] for row in rows], ddof=1))
                                  if state == "prediction" and len(rows) > 1 else None),
                } for metric in STRUCTURE_METRICS
                    if not (state == "reference" and metric == "macro_average_precision")}
                for state in ("reference", "mixed", "prediction")}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--cells", type=Path, required=True, help="Identity means exported by PhysicalCellAccumulator")
    args = parser.parse_args()
    with np.load(args.cells, allow_pickle=False) as bank:
        cells = {key: bank[key] for key in ("ids", "genes", "clean", "mixed", "prediction")}
    print(json.dumps(evaluate_cells(cells, args.split), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
