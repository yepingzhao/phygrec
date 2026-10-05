"""Validate checkpoints produced by the public PhyGRec implementation."""

from phygrec.protocol import ABLATIONS


TRAINING_KEYS = {"ablation", "solver_lr", "circle_lr", "aim_lr", "gcc_lr"}
CHECKPOINT_VERSION = 1


def release_hyperparameters(values: dict) -> dict:
    """Accept the public model parameters and Lightning serialization metadata."""
    values = dict(values)
    for name, expected in {
        "_class_path": "phygrec.training.module.PhyGRecModule",
        "_instantiator": "lightning.pytorch.cli.instantiate_module",
    }.items():
        if name in values and values.pop(name) != expected:
            raise ValueError(f"Unsupported checkpoint {name}")
    if set(values) - TRAINING_KEYS or values.get("ablation") not in ABLATIONS:
        raise ValueError("Checkpoint must use the public PhyGRec model parameters")
    return values
