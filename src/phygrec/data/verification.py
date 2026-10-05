"""Check released benchmark hashes, paths, and HDF5 schema."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

import h5py

from phygrec.data.hashing import sha256_file


from phygrec.protocol import DATA


def verify_scene_split_metadata(handle: h5py.File, split: str) -> None:
    for scene in handle["scenes"].values():
        if PurePosixPath(scene.attrs["candidate_shard"]).parent.name != split:
            raise ValueError(f"Scene split metadata mismatch: {handle.filename}/{scene.name}")


def verify_split_isolation() -> None:
    folds = {"main": None, "a9": "Y40360A9", "l7": "Y40360L7", "na": "Y40360NA"}
    for fold, heldout_chip in folds.items():
        directory = DATA / "benchmark" / (
            "main" if fold == "main" else f"loco/fold_{fold}"
        )
        labels = {}
        gene_panels = {}
        for split in ("train", "val", "test"):
            with h5py.File(directory / f"{split}.h5") as handle:
                verify_scene_split_metadata(handle, split)
                source_labels = handle["nodes/source_label"].asstr()[:].tolist()
                if len(source_labels) != len(set(source_labels)):
                    raise ValueError(f"Duplicate source identity: {fold}/{split}")
                labels[split] = set(source_labels)
                gene_panels[split] = tuple(handle["nodes/gene"].asstr()[:])
        if labels["train"] & labels["val"] or labels["train"] & labels["test"] or labels["val"] & labels["test"]:
            raise ValueError(f"Source identity crosses splits: {fold}")
        if len(set(gene_panels.values())) != 1 or gene_panels["train"] != tuple(
            (directory / "genes.txt").read_text().splitlines()
        ):
            raise ValueError(f"Gene panel mismatch: {fold}")
        if heldout_chip is not None:
            for split in ("train", "val"):
                if any(label.startswith(f"{heldout_chip}_") for label in labels[split]):
                    raise ValueError(f"Held-out chip enters {fold}/{split}")


def main() -> None:
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        path = DATA / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if item["type"] == "benchmark":
            with h5py.File(path) as handle:
                if handle.attrs["format"] != "dacc_shared_scene_graphs_v1":
                    raise ValueError(f"Unexpected store format: {path}")
                if len(handle["nodes/gene"]) != 1000:
                    raise ValueError(f"Unexpected gene count: {path}")
                if len(handle["index/scene_id"]) == 0:
                    raise ValueError(f"Empty scene store: {path}")
    verify_split_isolation()
    print(f"Verified {len(manifest['files'])} released files")


if __name__ == "__main__":
    main()
