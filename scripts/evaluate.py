"""Load a PhyGRec checkpoint and score the published test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phygrec.evaluation.expression import evaluate_model
from phygrec import ROOT, SEEDS, SPLITS, VARIANTS
from phygrec.training.validation import load_best_checkpoint


def evaluate(split: str, seed: int, device_name: str,
             checkpoint_path: Path | None = None, *, structure: bool = False,
             variant: str | None = None) -> dict:
    from phygrec.training.module import PhyGRecModule

    checkpoint = checkpoint_path or ROOT / load_best_checkpoint(split, seed, variant=variant)["checkpoint"]
    model = PhyGRecModule.load_from_checkpoint(str(checkpoint), map_location="cpu").to(device_name).eval()
    return evaluate_model(model, split, seed, device_name, structure=structure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint in the public PhyGRec format")
    parser.add_argument("--variant", choices=VARIANTS, help="Use the checkpoint selected during ablation training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--structure", action="store_true", help="Also score physical-cell annotation and clustering")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.split, args.seed, args.device, args.checkpoint,
                             structure=args.structure, variant=args.variant), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
