"""Load released shared-scene graph stores for training and evaluation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


FORMAT = "dacc_shared_scene_graphs_v1"


_NODE_TENSOR_KEYS = ("clean", "initial", "train_mask")
_OBSERVATION_TENSOR_KEYS = ("distance", "candidate_mask")


def pack_scene_graphs(graphs: list[dict]) -> dict:
    """Pack independent scenes into one block-diagonal graph.

    No cross-scene edges are introduced: every source/donor index is shifted by
    its scene's node offset.  The packed graph is intentionally optimized as
    one receiver-cell batch: ordinary mean losses therefore weight every
    receiver observation equally.
    """
    if not graphs:
        raise ValueError("cannot pack an empty scene batch")
    genes = graphs[0]["genes"]
    if any(not np.array_equal(graph["genes"], genes) for graph in graphs[1:]):
        raise ValueError("packed scenes must use an identical gene panel")
    candidate_width = graphs[0]["donors"].shape[1]
    if any(graph["donors"].shape[1] != candidate_width for graph in graphs):
        raise ValueError("packed scenes must use an identical candidate width")

    node_counts = [int(graph["clean"].shape[0]) for graph in graphs]
    observation_counts = [int(graph["source"].shape[0]) for graph in graphs]
    node_offsets = np.cumsum([0, *node_counts[:-1]], dtype=np.int64).tolist()
    packed = {
        key: torch.cat([graph[key] for graph in graphs], dim=0)
        for key in _NODE_TENSOR_KEYS + _OBSERVATION_TENSOR_KEYS
    }
    packed["source"] = torch.cat([
        graph["source"] + offset for graph, offset in zip(graphs, node_offsets)
    ])
    packed["donors"] = torch.cat([
        graph["donors"] + offset for graph, offset in zip(graphs, node_offsets)
    ])
    device = packed["source"].device
    packed["observation_graph_index"] = torch.repeat_interleave(
        torch.arange(len(graphs), device=device),
        torch.as_tensor(observation_counts, device=device),
    )
    packed.update({
        "scene_id": [graph["scene_id"] for graph in graphs],
        "combination": [graph["combination"] for graph in graphs],
        "labels": np.concatenate([graph["labels"] for graph in graphs]),
    })
    return packed


class SharedSceneGraphDataset(torch.utils.data.Dataset):
    """Lazily localize one scene against a shared clean-expression matrix."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).resolve())
        with h5py.File(self.path, "r") as handle:
            if handle.attrs.get("format") != FORMAT:
                raise ValueError(f"unexpected shared scene format: {self.path}")
            self.scene_ids = handle["index/scene_id"].asstr()[:].tolist()
            self.labels = handle["nodes/source_label"].asstr()[:]
            self.genes = handle["nodes/gene"].asstr()[:]
            # One CPU copy is intentionally shared by all lazily read scenes.
            # HDF5 fancy indexing into a compressed 2-D dataset is orders of
            # magnitude slower than gathering these small (about 44 MB for the
            # paper data) arrays in memory.
            self.clean_raw = handle["nodes/clean_raw"][:]
        self._handle: h5py.File | None = None

    def _scene_handle(self) -> h5py.File:
        # DataLoader workers receive their own lazy handle after forking.  A
        # persistent handle avoids reopening the same 51 MB store per scene.
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, index: int) -> dict:
        scene_id = self.scene_ids[index]
        handle = self._scene_handle()
        group = handle[f"scenes/{scene_id}"]
        try:
            source_global = group["source_global"][:]
            exact_global = group["exact_donor_global"][:]
            exact_fraction = group["exact_fraction"][:]
            exact_mask = group["exact_mask"][:].astype(bool)
            candidate_global = group["candidate_donor_global"][:]
            candidate_mask = group["candidate_mask"][:].astype(bool)
            # The model-visible node set must be determined exclusively by
            # observable receivers and ordinary spatial candidates.  Exact
            # synthetic donors are used below to generate ``mixed`` directly
            # from the shared truth matrix, but must never make an otherwise
            # absent node appear in the localized graph.
            active = np.concatenate([
                source_global,
                candidate_global[candidate_mask],
            ])
            global_nodes = np.unique(active).astype(np.int64)
            distance = group["candidate_distance"][:]
            combination = str(group.attrs["combination"])
        finally:
            # Explicitly release the group proxy while retaining the worker's
            # read-only file handle.
            del group
        clean = self.clean_raw[global_nodes]
        global_to_local = np.full(len(self.labels), -1, dtype=np.int32)
        global_to_local[global_nodes] = np.arange(len(global_nodes), dtype=np.int32)
        source = global_to_local[source_global]
        candidate_safe = candidate_global.copy()
        candidate_safe[~candidate_mask] = 0
        donors = global_to_local[candidate_safe]
        mixed = self.clean_raw[source_global].copy()
        for slot in range(exact_global.shape[1]):
            valid = exact_mask[:, slot]
            if np.any(valid):
                weights = exact_fraction[valid, slot]
                exact_nodes = exact_global[valid, slot]
                mixed[valid] += self.clean_raw[exact_nodes] * weights[:, None]
        initial = np.full(clean.shape, np.inf, dtype=np.float32)
        observed = np.zeros(len(clean), dtype=bool)
        observed[source] = True
        if not np.any(observed):
            raise ValueError("graph has no observed source nodes")
        np.minimum.at(initial, source, mixed)
        initial[~observed] = np.median(initial[observed], axis=0).astype(np.float32)
        graph = {
            "scene_id": scene_id,
            "combination": combination,
            "clean": torch.from_numpy(clean).float(),
            "source": torch.from_numpy(source).long(),
            "donors": torch.from_numpy(donors).long(),
            "initial": torch.from_numpy(initial).float(),
            "train_mask": torch.from_numpy(observed).bool(),
            "labels": self.labels[global_nodes],
            "genes": self.genes,
            "distance": torch.from_numpy(distance).float(),
            "candidate_mask": torch.from_numpy(candidate_mask).bool(),
        }
        return graph
