"""The public configurations instantiate the final model and its ablations."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from phygrec.model import PhyGRecModule
from phygrec.operators import CircleOverlapOperator


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "main",
    "ablation/rb",
    "ablation/aim",
    "ablation/gcc",
)


@pytest.mark.parametrize("variant", VARIANTS)
def test_final_method_and_ablation_configurations(variant):
    config = yaml.safe_load((ROOT / "configs" / variant / "seed20260816.yaml").read_text())
    args = config["model"]
    model = PhyGRecModule(**args)
    assert isinstance(model.recovery.forward_operator, CircleOverlapOperator)
    assert (len(model.recovery.solver.graph_correctors) == 0) == (
        variant == "ablation/gcc"
    )
    assert (model.recovery.solver.ablation == "rb") == (
        variant == "ablation/rb"
    )
    assert (len(model.recovery.solver.adaptive_modulators) == 0) == (
        variant == "ablation/aim"
    )
    assert (model.recovery.solver.momentum_logits is None) == (
        variant == "ablation/aim"
    )


def test_cli_accepts_released_config():
    result = subprocess.run(
        [
            sys.executable, "-m", "phygrec.cli.main", "fit",
            "--config", "configs/main/seed20260816.yaml", "--print_config",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    printed = yaml.safe_load(result.stdout)
    expected = yaml.safe_load((ROOT / "configs/main/seed20260816.yaml").read_text())
    assert printed["model"] == expected["model"]


def test_cli_rejects_model_class_selection():
    result = subprocess.run(
        [sys.executable, "-m", "phygrec.cli.main", "fit", "--config",
         "configs/main/seed20260816.yaml", "--model.class_path",
         "phygrec.model.PhyGRecModule", "--print_config"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "class_path" in result.stderr
