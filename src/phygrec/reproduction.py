"""Run the published training, validation selection and test sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from phygrec.protocol import DATA, ROOT, SEEDS, SPLITS, VARIANTS
from phygrec.selection import select_checkpoint
from phygrec.structure_evaluation import STRUCTURE_METRICS


METRICS = ("count_mae", "log_mae", "relative_l1", "count_rmse", "log_rmse", "relative_l2")


def experiment_runs(experiment: str) -> list[tuple[str, str | None]]:
    runs = []
    if experiment in ("main", "ablation", "all"):
        runs.append(("main", None))
    if experiment in ("loco", "all"):
        runs.extend((split, None) for split in ("a9", "l7", "na"))
    if experiment in ("ablation", "all"):
        runs.extend(("main", variant) for variant in VARIANTS)
    if not runs:
        raise ValueError(f"Unknown experiment: {experiment}")
    return runs


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
                   for metric in METRICS},
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
                for metric in METRICS
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


def run(experiment: str, seeds: list[int], *, evaluate_only: bool = False,
        dry_run: bool = False, device: str | None = None, structure: bool = False) -> dict | list[dict]:
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each seed must occur once")
    plan = []
    for split, variant in experiment_runs(experiment):
        group = f"ablation/{variant}" if variant else split
        for seed in seeds:
            if seed not in SEEDS:
                raise ValueError(f"Unknown seed: {seed}")
            config = f"configs/{group}/seed{seed}.yaml"
            selection_args = ["--split", split, "--seed", str(seed)]
            if variant:
                selection_args += ["--variant", variant]
            commands = []
            if not evaluate_only:
                commands.append([sys.executable, "-m", "phygrec.cli.main", "fit", "--config", config])
            checkpoint = str(DATA / f"checkpoints/{split}/seed{seed}.ckpt") if evaluate_only and not variant else "<val-selected checkpoint>"
            if checkpoint == "<val-selected checkpoint>":
                commands.append([sys.executable, "scripts/select_checkpoint.py", *selection_args])
            commands.append([sys.executable, "scripts/evaluate.py", "--split", split,
                             "--seed", str(seed), "--checkpoint", checkpoint])
            if structure:
                commands[-1].append("--structure")
            plan.append({"split": split, "seed": seed, "variant": variant,
                         "config": config, "commands": commands})
    if dry_run:
        return plan

    import torch
    from phygrec.evaluation import evaluate

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    output = ROOT / "runs/results"
    for item in plan:
        split, seed, variant = item["split"], item["seed"], item["variant"]
        if not evaluate_only:
            subprocess.run(item["commands"][0], cwd=ROOT, check=True)
        if evaluate_only and not variant:
            selected = {"checkpoint": str(DATA / f"checkpoints/{split}/seed{seed}.ckpt")}
        else:
            selected = select_checkpoint(split, seed, variant=variant)
        checkpoint = Path(selected["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        scores = evaluate(split, seed, device, checkpoint, structure=structure)
        result = {**scores, "variant": variant, "selection": selected}
        directory = output / (variant or split)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"seed{seed}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        results.append(result)
    summary = summarize_results(results)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{experiment}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("main", "loco", "ablation", "all"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--evaluate-only", action="store_true", help="Use bundled or previously val-selected checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan without training or evaluating")
    parser.add_argument("--device", help="Evaluation device; defaults to CUDA when available")
    parser.add_argument("--structure", action="store_true", help="Also run paper annotation and clustering scoring")
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.seeds, evaluate_only=args.evaluate_only,
                         dry_run=args.dry_run, device=args.device, structure=args.structure), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
