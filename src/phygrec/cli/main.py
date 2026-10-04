"""LightningCLI entry point for the final PhyGRec model."""

from lightning.pytorch.cli import LightningCLI

from phygrec.data.graph_datamodule import GraphDataModule
from phygrec.model import PhyGRecModule


def main() -> None:
    LightningCLI(
        model_class=PhyGRecModule,
        datamodule_class=GraphDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
