from __future__ import annotations


import h5py
import numpy as np
import torch

from phygrec.data.graph_datamodule import (
    SceneBatchSampler,
    GraphDataModule,
)
from phygrec.data.shared_scene_graphs import (
    FORMAT,
    SharedSceneGraphDataset,
    pack_scene_graphs,
)
from phygrec.models.inputs import build_compliant_model_input
from phygrec.models.solver import PhyGRecSolver
from phygrec.training.losses import recovery_loss


def _write_store(path) -> None:
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = FORMAT
        nodes = handle.create_group("nodes")
        nodes.create_dataset(
            "clean_raw",
            data=np.asarray([[10, 0], [0, 4], [2, 2], [1, 3]], np.float32),
        )
        nodes.create_dataset("clean_total", data=np.asarray([10, 4, 4, 4], np.float32))
        nodes.create_dataset("area", data=np.asarray([100, 200, 300, 400], np.float32))
        nodes.create_dataset("spatial", data=np.arange(8, dtype=np.float32).reshape(4, 2))
        nodes.create_dataset(
            "source_label", data=np.asarray(["A_1", "B_1", "B_2", "C_1"], object),
            dtype=string,
        )
        nodes.create_dataset("gene", data=np.asarray(["g1", "g2"], object), dtype=string)
        scenes = handle.create_group("scenes")
        combinations = ["a9l7", "a9na", "l7na"]
        for number, combination in enumerate(combinations):
            group = scenes.create_group(f"scene_{number:06d}")
            group.attrs["combination"] = combination
            group.create_dataset("source_global", data=np.asarray([0], np.int32))
            group.create_dataset("exact_donor_global", data=np.asarray([[1]], np.int32))
            group.create_dataset("exact_fraction", data=np.asarray([[0.5]], np.float32))
            group.create_dataset("exact_mask", data=np.asarray([[1]], np.uint8))
            group.create_dataset(
                "candidate_donor_global", data=np.asarray([[1, 2]], np.int32)
            )
            group.create_dataset("candidate_distance", data=np.asarray([[1, 2]], np.float32))
            group.create_dataset("candidate_mask", data=np.asarray([[1, 1]], np.uint8))
            group.create_dataset(
                "candidate_relative_displacement",
                data=np.asarray([[[1, 0], [2, 0]]], np.float32),
            )
            group.create_dataset(
                "candidate_target_fraction", data=np.asarray([[0.5, 0]], np.float32)
            )
            group.create_dataset(
                "candidate_target_edge", data=np.asarray([[1, 0]], np.uint8)
            )
        index = handle.create_group("index")
        index.create_dataset(
            "scene_id",
            data=np.asarray([f"scene_{i:06d}" for i in range(3)], object),
            dtype=string,
        )
        index.create_dataset("combination", data=np.asarray(combinations, object), dtype=string)


def test_shared_scene_dataset_reconstructs_mixed_and_localizes_nodes(tmp_path):
    path = tmp_path / "shared.h5"
    _write_store(path)

    graph = SharedSceneGraphDataset(path)[0]

    assert graph["combination"] == "a9l7"
    assert graph["labels"].tolist() == ["A_1", "B_1", "B_2"]
    assert torch.equal(graph["source"], torch.tensor([0]))
    assert torch.equal(graph["donors"], torch.tensor([[1, 2]]))
    assert torch.allclose(graph["initial"][graph["source"]], torch.tensor([[10.0, 2.0]]))
    assert torch.equal(graph["train_mask"], torch.tensor([True, False, False]))
    assert graph["candidate_mask"].all()
    assert "candidate_target_edge" not in graph
    assert "node_area" not in graph


def test_repeated_receiver_observations_use_gene_wise_minimum(tmp_path):
    path = tmp_path / "shared.h5"
    _write_store(path)
    with h5py.File(path, "r+") as handle:
        scene = handle["scenes/scene_000000"]
        replacements = {
            "source_global": np.array([0, 0], np.int32),
            "exact_donor_global": np.array([[1], [2]], np.int32),
            "exact_fraction": np.full((2, 1), .5, np.float32),
            "exact_mask": np.ones((2, 1), np.uint8),
            "candidate_donor_global": np.array([[1, 2], [1, 2]], np.int32),
            "candidate_distance": np.array([[1, 2], [1, 2]], np.float32),
            "candidate_mask": np.ones((2, 2), np.uint8),
        }
        for name, value in replacements.items():
            del scene[name]
            scene.create_dataset(name, data=value)
    graph = SharedSceneGraphDataset(path)[0]
    # The observations are [10, 2] and [11, 1]; neither equals their minimum.
    assert torch.equal(graph["initial"][0], torch.tensor([10., 1.]))
    assert torch.equal(graph["initial"][1:], torch.tensor([[10., 1.], [10., 1.]]))


def test_hidden_exact_donor_does_not_change_model_visible_node_set(tmp_path):
    path = tmp_path / "shared.h5"
    _write_store(path)
    with h5py.File(path, "r+") as handle:
        scene = handle["scenes/scene_000000"]
        scene["exact_donor_global"][...] = np.asarray([[3]], np.int32)
        scene["exact_fraction"][...] = np.asarray([[0.5]], np.float32)

    graph = SharedSceneGraphDataset(path)[0]

    assert "C_1" not in graph["labels"].tolist()
    assert graph["labels"].tolist() == ["A_1", "B_1", "B_2"]
    assert torch.allclose(graph["initial"][graph["source"]], torch.tensor([[10.5, 1.5]]))




def test_training_batch_packs_three_scene_graphs(tmp_path):
    path = tmp_path / "shared.h5"
    _write_store(path)
    module = GraphDataModule(
        shared_scene_store=str(path),
        seed=20260816, num_workers=0, pin_memory=False, persistent_workers=False,
    )
    module.setup("fit")

    batch = next(iter(module.train_dataloader()))

    assert isinstance(batch, dict)
    assert len(batch["scene_id"]) == 3
    assert batch["source"].numel() == 3


def test_scene_batches_cover_each_scene_once_and_shuffle_by_epoch() -> None:
    sampler = SceneBatchSampler(363, seed=17)
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    assert set(map(len, batches)) == {3}
    assert sorted(flattened) == list(range(363))
    assert len(flattened) == len(set(flattened))
    sampler.set_epoch(1)
    assert [index for batch in sampler for index in batch] != flattened




def test_packed_scene_losses_equal_mean_of_independent_scene_losses(tmp_path):
    path = tmp_path / "shared.h5"
    _write_store(path)
    dataset = SharedSceneGraphDataset(path)
    graphs = [dataset[0], dataset[1]]
    for graph in graphs:
        for key in ("initial", "clean"):
            graph[key] = torch.nn.functional.pad(graph[key], (0, 998))
        graph["genes"] = np.array([f"g{i}" for i in range(1000)])
    packed = pack_scene_graphs(graphs)
    model = PhyGRecSolver().eval()

    separate_inverse = torch.stack([
        recovery_loss(
            model, graph,
            model_graph=build_compliant_model_input(graph),
        )[0]
        for graph in graphs
    ]).mean()
    packed_inverse = recovery_loss(
        model, packed,
        model_graph=build_compliant_model_input(packed),
    )[0]
    assert torch.allclose(packed_inverse, separate_inverse, atol=1e-6, rtol=1e-6)


def test_shared_scene_mode_keeps_nontrain_inputs_sealed(tmp_path):
    path = tmp_path / "shared.h5"
    _write_store(path)
    try:
        GraphDataModule(
            shared_scene_store=str(path),
            seed=20260816,
            test_path="test.h5",
        )
    except TypeError as error:
        assert "test_path" in str(error)
    else:
        raise AssertionError("shared scene mode accepted an unsealed test path")
