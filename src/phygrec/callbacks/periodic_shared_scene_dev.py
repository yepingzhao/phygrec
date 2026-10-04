"""In-Trainer exact micro-averaged evaluation for shared-scene dev stores."""

from __future__ import annotations

import json
from pathlib import Path
import platform

import torch
from lightning.pytorch.callbacks import Callback


from phygrec.data.hashing import sha256_file
from phygrec.evaluation import evaluate_dev
from phygrec.protocol import ROOT as REPOSITORY_ROOT


class PeriodicSharedSceneDevCallback(Callback):
    """Evaluate exact receiver-entry dev recovery without restarting Trainer."""

    def __init__(
        self,
        run_name: str,
        output_root: str,
        store_root: str,
        interval_epochs: int = 10,
        steps_per_epoch: int = 61,
        workers: int = 4,
    ) -> None:
        super().__init__()
        if not run_name:
            raise ValueError("run_name must be nonempty")
        if interval_epochs < 1 or steps_per_epoch < 1:
            raise ValueError("epoch interval and steps per epoch must be positive")
        self.run_name = run_name
        self.output_root = str(output_root)
        self.store_root = str(store_root)
        self.interval_epochs = int(interval_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.workers = int(workers)

    @property
    def run_root(self) -> Path:
        return REPOSITORY_ROOT / self.output_root / "runs" / self.run_name

    def _checkpoint_audit(
        self, path: Path, completed_epochs: int, expected_step: int
    ) -> dict:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        updates = raw.get("optimizer_update_count")
        if updates != expected_step:
            raise ValueError(f"periodic checkpoint update count mismatch: {updates}")
        # Lightning's ``global_step`` is an optimizer-loop implementation
        # detail under manual optimization and gradient accumulation.  The
        # module-owned counters are checkpointed beside the weights and count
        # actual optimizer updates, so use those as the fail-closed audit.
        lightning_global_step = int(raw.get("global_step", -1))
        return {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(path),
            "checkpoint_epoch_metadata": int(raw.get("epoch", -1)),
            "completed_epochs": completed_epochs,
            "expected_optimizer_updates": expected_step,
            "lightning_global_step": lightning_global_step,
            "optimizer_update_count": updates,
        }

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Epoch is the scheduling unit.  Do not infer it from Lightning's
        # global_step: accumulated manual-optimization runs can legitimately
        # expose a different global_step convention while still executing the
        # exact requested number of optimizer updates.
        completed_epochs = int(trainer.current_epoch) + 1
        expected_step = completed_epochs * self.steps_per_epoch
        actual_step = int(pl_module.optimizer_update_count.detach().cpu().item())
        if actual_step != expected_step:
            raise ValueError(
                "periodic dev optimizer-update mismatch: "
                f"epoch={completed_epochs}, expected={expected_step}, "
                f"actual={actual_step}, lightning_global_step={trainer.global_step}"
            )
        if completed_epochs % self.interval_epochs != 0:
            return
        score_path = self.run_root / f"dev_epoch_{completed_epochs:03d}.json"
        checkpoint_dir = self.run_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        snapshot = checkpoint_dir / f"epoch_{completed_epochs:03d}.ckpt"
        if snapshot.exists() or score_path.exists():
            raise FileExistsError(f"development point already exists: {score_path}, {snapshot}")
        trainer.save_checkpoint(snapshot)
        audit = self._checkpoint_audit(snapshot, completed_epochs, expected_step)
        was_training = pl_module.training
        pl_module.eval()
        try:
            dev = evaluate_dev(
                pl_module,
                REPOSITORY_ROOT / self.store_root,
                device=pl_module.device,
                workers=self.workers,
            )
        finally:
            pl_module.train(was_training)
        payload = {
            "protocol": "periodic_shared_scene_dev_intrainer_v1",
            "run_name": self.run_name,
            "training_schedule": {"total_epochs": int(trainer.max_epochs)},
            "selection_split": "dev_only",
            "validation_execution": "lightning_callback_same_trainer_process",
            "dev_interval_epochs": self.interval_epochs,
            "evaluation_batching": {
                "graph_batch_size": 6,
                "collate": "packed_disconnected_scene_graphs",
                "aggregation": "receiver_cell_scene_entry_micro",
            },
            "checkpoint": audit,
            "dev": dev,
            "device": str(pl_module.device),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        score_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "dev_micro/combined_recovery": float(
                        dev["overall"]["combined_recovery"]
                    ),
                },
                step=int(trainer.global_step),
            )
