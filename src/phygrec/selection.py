"""Select a PhyGRec checkpoint using development recovery only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


from phygrec.protocol import ROOT, SEEDS, SPLITS, VARIANTS


def select_checkpoint(
    split: str, seed: int, root: Path = ROOT, *, variant: str | None = None
) -> dict:
    if split not in SPLITS or seed not in SEEDS:
        raise ValueError("Unknown split or seed")
    if variant is not None and (split != "main" or variant not in VARIANTS):
        raise ValueError("Ablations require the main split and a known variant")

    run_group = Path("ablation") / variant if variant else Path(split)
    name = variant or ("full" if split == "main" else split)
    run = root / "runs" / run_group / f"seed{seed}" / "dev" / "runs" / f"phygrec_{name}_seed{seed}"
    scores = sorted(run.glob("dev_epoch_*.json"))
    if not scores:
        raise FileNotFoundError(f"No development scores found under {run}")

    candidates = []
    for path in scores:
        report = json.loads(path.read_text())
        if report["selection_split"] != "dev_only":
            raise ValueError(f"Invalid selection split: {path}")
        epoch = int(report["checkpoint"]["completed_epochs"])
        score = float(report["dev"]["overall"]["combined_recovery"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite development score: {path}")
        checkpoint = root / report["checkpoint"]["path"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        candidates.append((score, epoch, checkpoint, path))

    score, epoch, checkpoint, source = min(candidates, key=lambda row: (-row[0], row[1]))
    result = {
        "seed": seed,
        "selected_epoch": epoch,
        "dev_combined_recovery": score,
        "checkpoint": str(checkpoint.relative_to(root)),
        "score_report": str(source.relative_to(root)),
    }
    if variant is not None:
        return {"variant": variant, **result}
    return {"split": split, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="main")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(select_checkpoint(args.split, args.seed, variant=args.variant), indent=2))


if __name__ == "__main__":
    main()
