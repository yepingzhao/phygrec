"""Released scene metadata must agree with the benchmark split."""

import h5py
import pytest

from phygrec.verification import verify_scene_split_metadata


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_scene_metadata_matches_split(tmp_path, split):
    with h5py.File(tmp_path / f"{split}.h5", "w") as handle:
        scene = handle.create_group("scenes/scene_000000")
        scene.attrs["candidate_shard"] = f"units/a9l7/{split}/scene.csv.gz"
        verify_scene_split_metadata(handle, split)


def test_validation_metadata_cannot_name_a_different_split(tmp_path):
    with h5py.File(tmp_path / "val.h5", "w") as handle:
        scene = handle.create_group("scenes/scene_000000")
        scene.attrs["candidate_shard"] = "units/a9l7/train/scene.csv.gz"
        with pytest.raises(ValueError, match="Scene split metadata mismatch"):
            verify_scene_split_metadata(handle, "val")
