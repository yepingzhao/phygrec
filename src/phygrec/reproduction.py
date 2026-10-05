"""Run the published training, validation selection and test sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


from phygrec.protocol import DATA, ROOT, SEEDS, VARIANTS
from phygrec.selection import select_checkpoint
from phygrec.results import summarize_results


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
