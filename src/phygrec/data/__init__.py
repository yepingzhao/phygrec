"""Published benchmark paths and scene-graph loading."""

from pathlib import Path

from phygrec import DATA, SPLITS


def benchmark_directory(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return DATA / ("benchmark/main" if split == "main" else f"benchmark/loco/fold_{split}")
