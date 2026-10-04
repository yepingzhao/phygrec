"""Sealed shared-scene data loading for final PhyGRec training."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Sampler
from lightning import LightningDataModule

from phygrec.data.shared_scene_graphs import SharedSceneGraphDataset, pack_scene_graphs


class SceneBatchSampler(Sampler[list[int]]):
    """Use the published epoch shuffle and distribute scene counts evenly.

    The main split partitions 363 scenes into 121 three-scene batches.
    Other split sizes retain every scene and spread any remainder evenly.
    """

    def __init__(
        self, dataset_size: int, *, seed: int
    ) -> None:
        if dataset_size < 1:
            raise ValueError("even scene batching requires a nonempty dataset")
        self.dataset_size = int(dataset_size)
        self.batch_size = 3
        self.seed = int(seed)
        self.epoch = 0
        # Lightning looks through this attribute when setting sampler epochs.
        self.sampler = self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * self.epoch
        )
        indices = torch.randperm(
            self.dataset_size, generator=generator
        ).tolist()
        batch_count = len(self)
        base, extra = divmod(self.dataset_size, batch_count)
        start = 0
        for batch_index in range(batch_count):
            size = base + int(batch_index < extra)
            yield indices[start:start + size]
            start += size


class GraphDataModule(LightningDataModule):
    """Load the training store; keep development and test stores sealed."""

    def __init__(
        self,
        shared_scene_store: str,
        seed: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        if not shared_scene_store:
            raise ValueError("shared_scene_store is required")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        self.save_hyperparameters()
        self.train_dataset: SharedSceneGraphDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "predict":
            raise RuntimeError("prediction uses scripts/evaluate.py")
        self.train_dataset = SharedSceneGraphDataset(self.hparams.shared_scene_store)

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        if dataset is None:
            raise RuntimeError("call setup() before train_dataloader()")
        workers = int(self.hparams.num_workers)
        return DataLoader(
            dataset,
            batch_sampler=SceneBatchSampler(
                len(dataset),
                seed=int(self.hparams.seed),
            ),
            collate_fn=pack_scene_graphs,
            num_workers=workers,
            pin_memory=bool(self.hparams.pin_memory),
            persistent_workers=bool(self.hparams.persistent_workers) if workers else False,
        )

