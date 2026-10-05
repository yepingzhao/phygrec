"""Summarize published expression and structure metrics across training seeds."""

import numpy as np

from phygrec.protocol import SPLITS, VARIANTS


EXPRESSION_METRICS = ("count_mae", "log_mae", "relative_l1", "count_rmse", "log_rmse", "relative_l2")
STRUCTURE_METRICS = (
    "accuracy", "macro_precision", "balanced_accuracy", "macro_f1", "mcc",
    "cohen_kappa", "macro_average_precision", "source_agreement_ari",
    "source_agreement_nmi", "leiden_stability_ari", "frozen_label_silhouette_20pc",
    "leiden_clusters",
)


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
