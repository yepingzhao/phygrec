"""Aggregate receiver observations against the frozen physical-cell cohorts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from phygrec.data.hashing import sha256_file
from phygrec.protocol import SPLITS


RESOURCES = Path(__file__).parents[1] / "resources/structure"


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
            [sys.executable, "-m", "phygrec.evaluation.structure", "--split", split, "--cells", str(path)],
            env=environment, capture_output=True, text=True, check=True,
        )
    return json.loads(completed.stdout)


def evaluate_cells(cells: dict, split: str) -> dict:
    """Score only the frozen nonzero Reference cohort, after prediction."""
    from phygrec.evaluation.structure_metrics import (
        CLASS_ORDER, annotation_metric_record, annotation_protocol,
        clustering_metrics, independent_pca,
    )

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
            "reference_expression_sha256": metadata["clean_sha256"], "states": scores}


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
