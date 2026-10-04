"""Development-only selection breaks score ties at the earliest epoch."""

from __future__ import annotations

import json

from phygrec.selection import select_checkpoint


def test_select_checkpoint_uses_dev_score_and_earliest_tie(tmp_path):
    run = tmp_path / "runs/ablation/gcc/seed20260816/dev/runs/phygrec_gcc_seed20260816"
    run.mkdir(parents=True)
    for epoch, score in [(10, 0.8), (20, 0.9), (30, 0.9)]:
        checkpoint = run / "checkpoints" / f"epoch_{epoch:03d}.ckpt"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.touch()
        report = {
            "selection_split": "dev_only",
            "checkpoint": {
                "completed_epochs": epoch,
                "path": str(checkpoint.relative_to(tmp_path)),
            },
            "dev": {"overall": {"combined_recovery": score}},
        }
        (run / f"dev_epoch_{epoch:03d}.json").write_text(json.dumps(report))
    exploratory = run.parent / "exploratory_attempt"
    exploratory.mkdir()
    report["dev"]["overall"]["combined_recovery"] = 1.0
    (exploratory / "dev_epoch_030.json").write_text(json.dumps(report))
    selected = select_checkpoint("main", 20260816, tmp_path, variant="gcc")
    assert selected["selected_epoch"] == 20
    assert selected["dev_combined_recovery"] == 0.9


def test_main_and_loco_use_the_same_dev_selection(tmp_path):
    for split in ("main", "a9"):
        name = "full" if split == "main" else split
        run = tmp_path / "runs" / split / "seed20260816" / "dev/runs" / f"phygrec_{name}_seed20260816"
        run.mkdir(parents=True)
        for epoch, score in ((10, 0.8), (20, 0.9), (30, 0.9)):
            checkpoint = run / "checkpoints" / f"epoch_{epoch:03d}.ckpt"
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.touch()
            report = {
                "selection_split": "dev_only",
                "checkpoint": {
                    "completed_epochs": epoch,
                    "path": str(checkpoint.relative_to(tmp_path)),
                },
                "dev": {"overall": {"combined_recovery": score}},
            }
            (run / f"dev_epoch_{epoch:03d}.json").write_text(json.dumps(report))
        selected = select_checkpoint(split, 20260816, tmp_path)
        assert selected["split"] == split
        assert selected["selected_epoch"] == 20
        assert selected["dev_combined_recovery"] == 0.9
