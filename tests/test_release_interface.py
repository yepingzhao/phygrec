"""Reject exploration configurations and preserve release checkpoint loading."""

import pytest
import torch

from phygrec.checkpoints import release_hyperparameters
from phygrec.model import PhyGRecModule


@pytest.mark.parametrize("ablation,count", [
    ("none", 2832681), ("rb", 2832681), ("aim", 2620485), ("gcc", 212229),
])
def test_release_model_round_trip(tmp_path, ablation, count):
    model = PhyGRecModule(ablation=ablation)
    assert sum(p.numel() for p in model.parameters()) == count
    assert len(model.recovery.solver.step_logits) == 4
    model._update_ema()
    checkpoint = {"hyper_parameters": dict(model.hparams), "state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    loaded = PhyGRecModule.load_from_checkpoint(path)
    assert loaded.hparams.ablation == ablation
    assert all(torch.equal(value, loaded.state_dict()[name])
               for name, value in model.state_dict().items())
    assert all(torch.equal(value, loaded._ema[name]) for name, value in model._ema.items())


@pytest.mark.parametrize("field,value", [
    ("graph_corrector_hidden_dim", 256), ("iterations", 8), ("robust_residual", False),
    ("inverse_ema_decay", 0.0), ("post_gcc_step", True), ("operator_lr", 0.001),
])
def test_exploration_checkpoint_is_rejected(field, value):
    values = {"ablation": "none", field: value}
    with pytest.raises(ValueError):
        release_hyperparameters(values)


def test_unpublished_branch_combination_is_rejected():
    with pytest.raises(ValueError, match="public PhyGRec model parameters"):
        release_hyperparameters({"learned_preconditioner": False, "graph_corrector": False})
    with pytest.raises(ValueError, match="Unknown ablation"):
        PhyGRecModule(ablation="rb+gcc")


def test_exploration_model_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="public PhyGRec model parameters"):
        release_hyperparameters({"learned_preconditioner": True, "graph_corrector": True})
    model = PhyGRecModule()
    checkpoint = {"hyper_parameters": dict(model.hparams), "state_dict": model.state_dict()}
    with pytest.raises(ValueError, match="trained with the public PhyGRec"):
        model.on_load_checkpoint(checkpoint)


def test_training_cannot_resume_a_different_ablation():
    model = PhyGRecModule()
    checkpoint = {"hyper_parameters": {"ablation": "rb"}, "state_dict": model.state_dict()}
    with pytest.raises(ValueError, match="ablation does not match"):
        model.on_load_checkpoint(checkpoint)


def test_lightning_cli_metadata_is_not_a_model_parameter():
    values = {"ablation": "none", "_class_path": "phygrec.model.PhyGRecModule",
              "_instantiator": "lightning.pytorch.cli.instantiate_module"}
    assert release_hyperparameters(values) == {"ablation": "none"}
    values["_class_path"] = "other.Model"
    with pytest.raises(ValueError, match="Unsupported checkpoint _class_path"):
        release_hyperparameters(values)
