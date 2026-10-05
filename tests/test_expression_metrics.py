"""Expression scoring weights receiver entries, including uneven batches."""

import pytest
import torch

from phygrec.evaluation.expression_metrics import ExpressionMetricAccumulator


def test_receiver_weighting_and_rmse_are_independent_of_batch_partition():
    clean = torch.tensor([[1., 2.], [3., 4.], [2., 6.]])
    prediction = clean + torch.tensor([[0., 0.], [1., -1.], [2., 0.]])
    initial = clean + 3
    together = ExpressionMetricAccumulator()
    together.update(initial, prediction, clean)
    divided = ExpressionMetricAccumulator()
    for start, end in ((0, 1), (1, 3)):
        divided.update(initial[start:end], prediction[start:end], clean[start:end])
    scores = together.summary()
    assert scores == pytest.approx(divided.summary())
    assert scores["pred_raw_mae"] == pytest.approx(4 / 6)
    assert scores["pred_raw_l1"] == pytest.approx((0 + 2 / 7 + 2 / 8) / 3)
    assert scores["count_rmse"] == pytest.approx(1)
    assert scores["relative_l2"] == pytest.approx((0 + 2 ** .5 / 5 + 2 / 40 ** .5) / 3)


def test_relative_l2_uses_a_per_receiver_reference_norm_floored_at_one():
    totals = ExpressionMetricAccumulator()
    clean = torch.tensor([[0., 0.], [.3, .4]])
    prediction = clean + torch.tensor([[3., 4.], [.3, .4]])
    totals.update(clean, prediction, clean)
    assert totals.summary()["relative_l2"] == pytest.approx((5 + .5) / 2)


def test_empty_evaluation_cannot_produce_a_selection_score():
    with pytest.raises(ValueError, match="Empty evaluation"):
        ExpressionMetricAccumulator().summary()
