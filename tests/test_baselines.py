"""Frozen baseline architectures, target isolation and end-to-end selection."""

import pytest
import torch

from phygrec.baselines import runner
from phygrec.baselines.physical import FittedCircleOperator, physical_solve
from phygrec.data.shared_scene_graphs import pack_scene_graphs
from test_shared_scene_graphs import _write_store


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph():
    initial = torch.rand(6, 1000) + 0.1
    return {
        "initial": initial, "mixed": initial[:4].clone(),
        "source": torch.arange(4),
        "donors": torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]]),
        "distance": torch.tensor([[10., 30.]]).repeat(4, 1),
        "candidate_mask": torch.ones(4, 2, dtype=torch.bool),
        "observation_graph_index": torch.zeros(4, dtype=torch.long),
        "clean": initial * 0.9,
        "train_mask": torch.tensor([True, True, True, True, False, False]),
    }


@pytest.mark.parametrize("method,count", [
    ("gatv2", 2836968), ("graphsage", 2064360), ("gat", 7826920),
    ("gcn", 1553896), ("mpnn", 6777256), ("vae", 1604648), ("physical_pgd", 4),
])
def test_final_models_are_target_free_and_have_finite_gradients(method, count):
    model = runner.BaselinePrediction(method)
    assert sum(p.numel() for p in model.parameters()) == count
    batch = graph()
    model.eval()
    first = model(batch)
    poisoned = {**batch, "clean": torch.full_like(batch["clean"], float("nan"))}
    assert torch.equal(first, model(poisoned))
    assert first.shape == batch["initial"].shape
    assert torch.isfinite(first).all() and (first >= 0).all()
    if method != "physical_pgd":
        model.train()
        loss = runner.objective(model, model(batch), batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_physical_fit_and_diagonal_pgd_use_only_fixed_operator():
    operator = FittedCircleOperator()
    loss = runner.physical_fit_loss(operator, graph())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in operator.parameters())
    batch = graph()
    batch["mixed"] = batch["initial"][batch["source"]].clone()
    prediction, _ = physical_solve(batch, torch.zeros_like(batch["distance"]), .1, 100)
    assert torch.equal(prediction, batch["initial"])


def test_configs_cannot_silently_change_the_recorded_architecture(tmp_path, monkeypatch):
    directory = tmp_path / "configs/baselines"
    directory.mkdir(parents=True)
    (directory / "gatv2.yaml").write_text("hidden: 512\nn_layers: 2\nheads: 4\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="fixed published model"):
        runner.load_config("gatv2")


def test_training_selects_earliest_validation_tie_and_test_is_separate(tmp_path, monkeypatch):
    directory = tmp_path / "benchmark"
    directory.mkdir()
    # This small store exercises the real loader and loop; the toy model uses two genes.
    for split in ("train", "val"):
        _write_store(directory / f"{split}.h5")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "benchmark_directory", lambda _: directory)
    config = {"lr": .001, "weight_decay": 0., "batch_size": 3,
              "max_epochs": 2, "eval_every": 1, "gradient_clip": 5.}
    monkeypatch.setattr(runner, "load_config", lambda _: config)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.9))
            self.method = "gcn"

        def forward(self, batch):
            return self.scale * batch["initial"]

    monkeypatch.setattr(runner, "BaselinePrediction", lambda _: Toy())
    scores = iter([{"combined_recovery": .5}, {"combined_recovery": .5}])
    monkeypatch.setattr(runner, "validation_score", lambda *_: next(scores))
    checkpoint = runner.train("gcn", "main", runner.SEEDS[0], "cpu", 0)
    raw = torch.load(checkpoint, weights_only=True)
    assert raw["epoch"] == 1 and raw["selection_split"] == "val_only"
    assert not (directory / "test.h5").exists()
    assert len((checkpoint.parent / "history.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        runner.train("gcn", "main", runner.SEEDS[0], "cpu", 0)


def test_observation_expression_survives_scene_packing(tmp_path):
    path = tmp_path / "store.h5"
    _write_store(path)
    dataset = runner.SharedSceneGraphDataset(path)
    graphs = [dataset[0], dataset[1]]
    batch = pack_scene_graphs(graphs)
    assert torch.equal(batch["mixed"], torch.cat([g["mixed"] for g in graphs]))
