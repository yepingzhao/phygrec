"""Test the public evaluation boundary with caller settings that allow drift."""
import os

import pytest
import torch

from phygrec.evaluation import expression


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_enforces_and_restores_deterministic_execution(monkeypatch, fail):
    batch = {
        "initial": torch.tensor([[1., 2.], [3., 4.]]),
        "clean": torch.tensor([[1., 2.], [3., 4.]]),
        "source": torch.arange(2), "donors": torch.tensor([[1], [0]]),
        "distance": torch.ones(2, 1), "candidate_mask": torch.ones(2, 1, dtype=torch.bool),
        "labels": ["chip_0", "chip_1"], "scene_id": ["scene"],
        "observation_graph_index": torch.zeros(2, dtype=torch.long),
    }
    monkeypatch.setattr(expression, "SharedSceneGraphDataset", lambda path: [batch])
    monkeypatch.setattr(expression, "pack_scene_graphs", lambda scenes: scenes[0])
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def predictor(graph):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_num_threads() == 1
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        if fail:
            raise RuntimeError("evaluation probe failed")
        return graph["initial"]

    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark)
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if fail:
            with pytest.raises(RuntimeError, match="evaluation probe failed"):
                expression.evaluate_model(predictor, "main", 20260816, "cpu")
        else:
            result = expression.evaluate_model(predictor, "main", 20260816, "cpu")
            assert result["relative_l1"] == result["log_mae"] == 0
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_num_threads() == 2
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.set_num_threads(previous[2])
        torch.backends.cuda.matmul.allow_tf32 = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]
        torch.backends.cudnn.deterministic = previous[5]
        torch.backends.cudnn.benchmark = previous[6]
