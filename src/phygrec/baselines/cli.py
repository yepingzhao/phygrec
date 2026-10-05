"""Command-line dispatch for the seven published comparison baselines."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from phygrec.baselines.runner import METHODS, train, evaluate
from phygrec.protocol import ROOT, SEEDS, SPLITS
from phygrec.experiments.results import summarize_results


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


if __name__ == "__main__":
    main()
