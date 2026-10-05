"""Paper aggregation uses seeds as replicates and equal LOCO fold weights."""

import pytest

from phygrec.reproduction import METRICS, run, summarize_results


def row(split, seed, value):
    return {"split": split, "seed": seed, **dict.fromkeys(METRICS, value)}


def test_loco_averages_folds_before_sample_standard_valiation():
    results = [row(split, seed, value)
               for seed, values in ((1, (0, 3, 6)), (2, (6, 9, 12)), (3, (12, 15, 18)))
               for split, value in zip(("a9", "l7", "na"), values)]
    summary = summarize_results(results)["loco"]
    assert summary["n_seeds"] == 3
    assert summary["metrics"]["log_mae"] == {"mean": 9., "sample_sd": 6.}


def test_partial_loco_cannot_silently_change_fold_weights():
    with pytest.raises(ValueError, match="all three folds"):
        summarize_results([row("a9", 1, 0), row("l7", 1, 3)])


@pytest.mark.parametrize("split,variant", [
    ("development_probe", None), ("main", "gain_only"), ("a9", "rb"),
])
def test_unpublished_result_identity_is_rejected(split, variant):
    with pytest.raises(ValueError, match="result identity"):
        summarize_results([{**row(split, 1, 0), "variant": variant}])


def test_loco_ablation_cannot_overwrite_the_full_fold():
    results = [row(split, 1, value)
               for split, value in zip(("a9", "l7", "na"), (1, 2, 3))]
    results.append({**row("a9", 1, 100), "variant": "rb"})
    with pytest.raises(ValueError, match="result identity"):
        summarize_results(results)


def test_ablation_plan_includes_full_model_and_all_three_removals():
    plan = run("ablation", [20260816], dry_run=True)
    assert len(plan) == 4
    assert plan[0]["variant"] is None
    assert len({item["config"] for item in plan}) == 4
    for item in plan:
        assert len(item["commands"]) == 3
        assert "fit" in item["commands"][0]
        assert item["commands"][1][1] == "scripts/select_checkpoint.py"
        assert item["commands"][2][1] == "scripts/evaluate.py"
