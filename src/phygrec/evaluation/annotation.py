"""Cell-type annotation using the frozen Reference-marker programs."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize

from phygrec.utils import normalize_log1p


MARKER_PROGRAMS = {
    "T cells": ("CD3D", "CD4", "CD8A", "CD8B", "IL7R", "CCR7", "TCF7", "LEF1"),
    "NK cells": ("NKG7", "GNLY", "KLRD1", "NCR1", "KLRF1", "FCGR3A", "XCL1"),
    "B cells": ("CD79A", "CD19", "MS4A1", "CD74", "CD27", "TCL1A"),
    "Monocytes": (
        "LYZ",
        "FCN1",
        "CTSD",
        "CD14",
        "S100A8",
        "S100A9",
        "S100A12",
        "VCAN",
        "FCGR3A",
    ),
    "Platelets": ("PPBP", "PF4", "GP9", "TUBB1"),
}


CLASS_ORDER = tuple(MARKER_PROGRAMS)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def marker_scores(
    expression: np.ndarray,
    *,
    scaler: StandardScaler,
    marker_indices: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply the frozen Reference-fitted marker annotator up to continuous scores."""
    scaled = scaler.transform(normalize_log1p(expression)).astype(
        np.float32, copy=False
    )
    # Fixed numerical safeguard applied identically to every expression state.
    np.clip(scaled, -10.0, 10.0, out=scaled)
    return np.column_stack(
        [scaled[:, marker_indices[name]].mean(axis=1) for name in CLASS_ORDER]
    )


def labels_from_scores(scores: np.ndarray) -> np.ndarray:
    return np.asarray(
        [CLASS_ORDER[index] for index in scores.argmax(axis=1)], dtype=str
    )


def annotation_protocol(
    source: np.ndarray,
    genes: np.ndarray,
) -> tuple[
    StandardScaler,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    str,
]:
    gene_index = {gene: index for index, gene in enumerate(genes)}
    marker_indices = {}
    for name, markers in MARKER_PROGRAMS.items():
        present = [gene for gene in markers if gene in gene_index]
        if len(present) != len(markers):
            raise ValueError(f"fixed panel lacks published markers for {name}: {present}")
        marker_indices[name] = np.asarray(
            [gene_index[gene] for gene in present], dtype=np.int64
        )

    scaler = StandardScaler().fit(normalize_log1p(source))
    source_scores = marker_scores(
        source,
        scaler=scaler,
        marker_indices=marker_indices,
    )
    reference_labels = labels_from_scores(source_scores)
    missing_classes = set(CLASS_ORDER) - set(reference_labels)
    if missing_classes:
        raise ValueError(
            "Reference per-cell marker reference omits classes: "
            f"{sorted(missing_classes)}"
        )
    encoded_reference = label_binarize(reference_labels, classes=list(CLASS_ORDER))
    marker_genes_used = ";".join(
        f"{name}:{len(marker_indices[name])}" for name in CLASS_ORDER
    )
    return scaler, reference_labels, encoded_reference, marker_indices, marker_genes_used


def annotation_metric_record(
    expression: np.ndarray,
    *,
    scaler: StandardScaler,
    reference_labels: np.ndarray,
    encoded_reference: np.ndarray,
    marker_indices: dict[str, np.ndarray],
) -> dict:
    scores = marker_scores(
        expression,
        scaler=scaler,
        marker_indices=marker_indices,
    )
    probabilities = softmax(scores)
    prediction = labels_from_scores(scores)
    row = {
        "predicted_classes": int(len(np.unique(prediction))),
        "accuracy": float(accuracy_score(reference_labels, prediction)),
        "macro_precision": float(
            precision_score(
                reference_labels,
                prediction,
                labels=list(CLASS_ORDER),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(reference_labels, prediction)
        ),
        "macro_f1": float(
            f1_score(
                reference_labels,
                prediction,
                labels=list(CLASS_ORDER),
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(reference_labels, prediction)),
        "cohen_kappa": float(cohen_kappa_score(reference_labels, prediction)),
        "macro_average_precision": float(
            average_precision_score(
                encoded_reference,
                probabilities,
                average="macro",
            )
        ),
    }
    for name in CLASS_ORDER:
        row[f"reference_{name}"] = int(np.sum(reference_labels == name))
        row[f"predicted_{name}"] = int(np.sum(prediction == name))
    row["confusion_matrix"] = confusion_matrix(reference_labels, prediction, labels=list(CLASS_ORDER)).tolist()
    return row
