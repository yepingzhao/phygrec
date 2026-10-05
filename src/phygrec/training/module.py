"""PhyGRec model and training module for the final four-Block protocol."""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path

import torch
from torch import Tensor

from lightning import LightningModule
from phygrec.models.inputs import build_compliant_model_input
from phygrec.models.solver import PhyGRecSolver
from phygrec.training.losses import recovery_loss
from phygrec.protocol import ABLATIONS, Ablation
from phygrec.training.checkpoints import CHECKPOINT_VERSION, release_hyperparameters


class PhyGRecModule(LightningModule):
    """The final four-Block PhyGRec model and its three ablations."""

    def __init__(
        self, ablation: Ablation = 'none',
        solver_lr: float = 0.003125, circle_lr: float = 0.06,
        aim_lr: float = 0.003, gcc_lr: float = 0.01,
    ) -> None:
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f'Unknown ablation: {ablation}')
        if any(not math.isfinite(rate) or rate <= 0 for rate in (solver_lr, circle_lr, aim_lr, gcc_lr)):
            raise ValueError('Learning rates must be positive')
        self.save_hyperparameters()
        self.recovery = PhyGRecSolver(ablation)
        self.register_buffer(
            'optimizer_update_count', torch.zeros((), dtype=torch.long), persistent=False
        )
        self._ema: dict[str, Tensor] = {}

    def forward(self, batch: dict) -> Tensor:
        with self._use_ema():
            return self.recovery(build_compliant_model_input(batch))

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        receiver_batch_size = int(batch["source"].numel())
        model_graph = build_compliant_model_input(batch)
        optimizer = self.optimizers()
        accumulation = 2
        if batch_idx % accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
        try:
            total_training_batches = int(self.trainer.num_training_batches)
        except (AttributeError, RuntimeError, TypeError):
            total_training_batches = -1
        accumulation_group_start = (batch_idx // accumulation) * accumulation
        accumulation_group_size = (
            min(accumulation, total_training_batches - accumulation_group_start)
            if total_training_batches > 0 else accumulation
        )
        solver_loss, solver_components = recovery_loss(
            self.recovery, batch, model_graph=model_graph
        )
        self.manual_backward(solver_loss / accumulation_group_size)
        last_batch = batch_idx + 1 == total_training_batches
        accumulation_end = (batch_idx + 1) % accumulation == 0 or last_batch
        gradient_norm = batch["initial"].new_zeros(())
        if accumulation_end:
            gradient_norm = self._clip_gradients(optimizer)
            optimizer.step()
            self.optimizer_update_count.add_(1)
            self._update_ema()
        self.log_dict(
            {f"train/solver_{k}": v for k, v in solver_components.items()},
            on_step=False, on_epoch=True, batch_size=receiver_batch_size,
        )
        self.log(
            "train/solver_loss", solver_loss, on_step=False, on_epoch=True,
            batch_size=receiver_batch_size,
        )
        self.log(
            "solver/lr", float(optimizer.param_groups[0]["lr"]),
            on_step=False, on_epoch=True, batch_size=receiver_batch_size,
        )
        self.log(
            "solver/gradient_norm_preclip", gradient_norm,
            on_step=False, on_epoch=True, batch_size=receiver_batch_size,
        )
        return solver_loss

    def _update_ema(self) -> None:
        decay = 0.99
        with torch.no_grad():
            for name, parameter in self.recovery.named_parameters():
                value = parameter.detach()
                if name not in self._ema:
                    self._ema[name] = value.clone()
                else:
                    # EMA tensors are intentionally serialized on CPU.  A
                    # resumed Lightning fit restores the checkpoint before
                    # moving the module to its accelerator, so this plain
                    # dictionary is not migrated by ``Module.to``.  Normalize
                    # it lazily against the live parameter before the first
                    # resumed update.
                    average = self._ema[name]
                    if average.device != value.device or average.dtype != value.dtype:
                        average = average.to(device=value.device, dtype=value.dtype)
                        self._ema[name] = average
                    average.mul_(decay).add_(
                        value, alpha=1.0 - decay
                    )

    @contextmanager
    def _use_ema(self):
        """Temporarily evaluate with EMA parameters, restoring live weights."""
        enabled = (
            not self.training
            and bool(self._ema)
        )
        if not enabled:
            yield
            return
        originals: dict[str, Tensor] = {}
        parameters = dict(self.recovery.named_parameters())
        with torch.no_grad():
            for name, average in self._ema.items():
                parameter = parameters[name]
                originals[name] = parameter.detach().clone()
                parameter.copy_(average.to(parameter))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, original in originals.items():
                    parameters[name].copy_(original)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["phygrec_checkpoint_version"] = CHECKPOINT_VERSION
        checkpoint["optimizer_update_count"] = int(self.optimizer_update_count.item())
        if self._ema:
            checkpoint["ema"] = {
                name: value.detach().cpu() for name, value in self._ema.items()
            }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        saved = release_hyperparameters(checkpoint["hyper_parameters"])
        if saved["ablation"] != self.hparams.ablation:
            raise ValueError("Checkpoint ablation does not match the configured model")
        if checkpoint.get("phygrec_checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("Checkpoint must be trained with the public PhyGRec implementation")
        self.optimizer_update_count.fill_(int(checkpoint["optimizer_update_count"]))
        self._ema = {
            name: value.clone()
            for name, value in checkpoint.get("ema", {}).items()
        }
        parameters = dict(self.recovery.named_parameters())
        if self._ema and (self._ema.keys() != parameters.keys() or any(
            average.shape != parameters[name].shape for name, average in self._ema.items()
        )):
            raise ValueError("Checkpoint EMA does not match the release architecture")

    def _clip_gradients(self, optimizer) -> Tensor:
        parameters = [parameter for group in optimizer.param_groups
                      for parameter in group["params"] if parameter.grad is not None]
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0).detach()

    def configure_optimizers(self):
        self.automatic_optimization = False
        solver = self.recovery.solver
        graph_parameters = list(solver.graph_correctors.parameters())
        aim_parameters = list(solver.adaptive_modulators.parameters())
        special_ids = {id(p) for p in graph_parameters + aim_parameters}
        core_parameters = [p for p in solver.parameters() if id(p) not in special_ids]
        groups = [{"params": core_parameters, "lr": float(self.hparams.solver_lr)}]
        if graph_parameters:
            groups.append({"params": graph_parameters, "lr": float(self.hparams.gcc_lr)})
        if aim_parameters:
            groups.append({"params": aim_parameters, "lr": float(self.hparams.aim_lr)})
        groups.append({
            "params": [self.recovery.output_residual_logit],
            "lr": float(self.hparams.solver_lr),
        })
        groups.append({
            "params": self.recovery.forward_operator.parameters(),
            "lr": float(self.hparams.circle_lr),
        })
        return torch.optim.AdamW(
            groups, lr=float(self.hparams.solver_lr),
            weight_decay=1e-4,
        )

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str | Path, map_location="cpu"):
        """Load a checkpoint trained with the public implementation."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        model = cls(**release_hyperparameters(checkpoint['hyper_parameters']))
        model.on_load_checkpoint(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return model
