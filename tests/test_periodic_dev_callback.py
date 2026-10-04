"""The published schedules audit optimizer updates independently of the loader."""

from types import SimpleNamespace
import hashlib

import pytest
import torch

from phygrec.callbacks.periodic_shared_scene_dev import PeriodicSharedSceneDevCallback
import phygrec.callbacks.periodic_shared_scene_dev as periodic_dev


@pytest.mark.parametrize("updates", [61, 21])
def test_update_audit_uses_protocol_count(updates: int) -> None:
    callback = PeriodicSharedSceneDevCallback(
        run_name="audit", output_root="runs/audit", store_root="data",
        interval_epochs=2, steps_per_epoch=updates,
    )
    trainer = SimpleNamespace(current_epoch=0, global_step=updates)
    module = SimpleNamespace(optimizer_update_count=torch.tensor(updates))
    callback.on_train_epoch_end(trainer, module)
    module.optimizer_update_count -= 1
    with pytest.raises(ValueError, match="optimizer-update mismatch"):
        callback.on_train_epoch_end(trainer, module)


def test_checkpoint_audit_records_file_digest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(periodic_dev, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "epoch_001.ckpt"
    torch.save({"optimizer_update_count": 61, "global_step": 122,
                "epoch": 0}, path)
    callback = PeriodicSharedSceneDevCallback(
        run_name="audit", output_root="runs/audit", store_root="data",
    )
    report = callback._checkpoint_audit(path, completed_epochs=1, expected_step=61)
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["completed_epochs"] == 1
    assert report["expected_optimizer_updates"] == 61
    assert report["lightning_global_step"] == 122


def test_sealed_dev_point_cannot_be_reused_from_matching_live_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(periodic_dev, "REPOSITORY_ROOT", tmp_path)
    callback = PeriodicSharedSceneDevCallback(
        run_name="audit", output_root="runs/audit", store_root="data",
        interval_epochs=1, steps_per_epoch=1,
    )
    checkpoint = callback.run_root / "checkpoints/epoch_001.ckpt"
    checkpoint.parent.mkdir(parents=True)
    state = {"weight": torch.tensor([1.0])}
    torch.save({"state_dict": state, "ema": {"weight": torch.tensor([2.0])}}, checkpoint)
    (callback.run_root / "dev_epoch_001.json").write_text("{}")
    # Matching live weights do not prove that EMA, data or the sealed score match.
    module = SimpleNamespace(optimizer_update_count=torch.tensor(1), state_dict=lambda: state)
    trainer = SimpleNamespace(current_epoch=0, global_step=1)
    with pytest.raises(FileExistsError, match="already exists"):
        callback.on_train_epoch_end(trainer, module)
