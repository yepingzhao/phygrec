"""Published experiment names, seeds and release-relative paths."""

from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEEDS = (20260816, 20260817, 20260818)
SPLITS = ("main", "a9", "l7", "na")
Ablation = Literal["none", "rb", "aim", "gcc"]
ABLATIONS = ("none", "rb", "aim", "gcc")
FOLDS = {
    "a9": ("Y40360A9", ("a9l7", "a9na")),
    "l7": ("Y40360L7", ("a9l7", "l7na")),
    "na": ("Y40360NA", ("a9na", "l7na")),
}
VARIANTS = (
    "rb",
    "aim",
    "gcc",
)
