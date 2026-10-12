"""Training selects checkpoints from validation, keeping the earliest score tie."""

import json
from types import SimpleNamespace

import pytest
import torch

import phygrec.training.validation as validation


@pytest.mark.parametrize("split,variant", [("main", None), ("a9", None), ("main", "gcc")])
def test_validation_records_best_checkpoint_and_preserves_it_on_resume(tmp_path, monkeypatch, split, variant):
    monkeypatch.setattr(validation, "REPOSITORY_ROOT", tmp_path)
    group = f"ablation/{variant}" if variant else split
    name = variant or ("full" if split == "main" else split)
    kwargs = dict(run_name=f"phygrec_{name}_seed20260816",
                  output_root=f"runs/{group}/seed20260816/val", store_root="benchmark",
                  interval_epochs=10, steps_per_epoch=1, workers=0)
    module = SimpleNamespace(optimizer_update_count=torch.tensor(0), training=True,
                             device=torch.device("cpu"))
    module.eval = lambda: setattr(module, "training", False)
    module.train = lambda mode: setattr(module, "training", mode)
    trainer = SimpleNamespace(max_epochs=100, logger=None)
    def save(path):
        torch.save({"optimizer_update_count": trainer.global_step,
                    "global_step": trainer.global_step, "epoch": trainer.current_epoch}, path)
    trainer.save_checkpoint = save
    for epoch, score, expected in ((10, .8, 10), (20, .9, 20), (30, .9, 20), (40, .7, 20)):
        # A new callback instance must retain the on-disk best record when resuming.
        callback = validation.PeriodicSharedSceneValidationCallback(**kwargs)
        trainer.current_epoch, trainer.global_step = epoch - 1, epoch
        module.optimizer_update_count.fill_(epoch)
        monkeypatch.setattr(validation, "evaluate_validation",
                            lambda *args, **kwargs: {"overall": {"combined_recovery": score}})
        callback.on_train_epoch_end(trainer, module)
        selected = validation.load_best_checkpoint(split, 20260816, tmp_path, variant=variant)
        assert selected["selected_epoch"] == expected
        assert selected["val_combined_recovery"] == (.8 if epoch == 10 else .9)
        assert module.training
        assert json.loads((tmp_path / selected["score_report"]).read_text())["selection_split"] == "val_only"
    assert not (tmp_path / "benchmark/test.h5").exists()


def test_evaluation_requires_a_record_selected_during_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="best.json"):
        validation.load_best_checkpoint("main", 20260816, tmp_path)


@pytest.mark.parametrize("split,seed,variant", [("validation", 20260816, None),
    ("main", 1, None), ("a9", 20260816, "rb"), ("main", 20260816, "gain_only")])
def test_unpublished_checkpoint_identity_is_rejected(tmp_path, split, seed, variant):
    with pytest.raises(ValueError):
        validation.load_best_checkpoint(split, seed, tmp_path, variant=variant)
