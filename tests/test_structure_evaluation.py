"""Frozen Reference scoring and physical-cell/fold/seed aggregation boundaries."""

import numpy as np
import pytest

pytest.importorskip("scanpy", reason="Install the structure extra for downstream scoring tests")
pytest.importorskip("leidenalg", reason="Install the structure extra for downstream scoring tests")

from phygrec.reproduction import METRICS, run, summarize_results
from phygrec.structure_evaluation import CellTotals, STRUCTURE_METRICS, load_reference
from phygrec.structure_metrics import (
    CLASS_ORDER, MARKER_PROGRAMS, annotation_metric_record, annotation_protocol,
    clustering_metrics, independent_pca,
)


def profiles():
    genes = np.asarray(list(dict.fromkeys(gene for markers in MARKER_PROGRAMS.values() for gene in markers)))
    generator = np.random.default_rng(42)
    clean = generator.uniform(.1, 1, (80, len(genes))).astype(np.float32)
    index = {gene: position for position, gene in enumerate(genes)}
    for group, markers in enumerate(MARKER_PROGRAMS.values()):
        clean[group * 16:(group + 1) * 16, [index[gene] for gene in markers]] += 20
    return clean, genes


def test_raw_identity_means_exclude_only_zero_reference(monkeypatch):
    reference = {"ids": np.asarray(["b", "a"]), "genes": np.asarray(["g1", "g2"])}
    monkeypatch.setattr("phygrec.structure_evaluation.load_reference", lambda *args: (reference, {}))
    total = CellTotals("main", reference["genes"])
    labels = np.asarray(["a", "b", "a", "zero"])
    clean = np.asarray([[2, 4], [3, 6], [2, 4], [0, 0]], dtype=np.float32)
    prediction = np.asarray([[1, 2], [3, 4], [9, 4], [1, 1]], dtype=np.float32)
    total.update(labels[:2], clean[:2], clean[:2], prediction[:2])
    total.update(labels[2:], clean[2:], clean[2:], prediction[2:])
    means = total.means()
    np.testing.assert_array_equal(means["ids"], ["b", "a"])
    np.testing.assert_array_equal(means["prediction"], [[3, 4], [5, 3]])
    with pytest.raises(ValueError, match="Nonzero receiver"):
        total.update(np.asarray(["unexpected"]), clean[:1], clean[:1], prediction[:1])


def test_missing_physical_cells_cannot_change_the_scored_cohort(monkeypatch):
    ref = {"ids": np.asarray(["a", "b"]), "genes": np.asarray(["g"])}
    monkeypatch.setattr("phygrec.structure_evaluation.load_reference", lambda *args: (ref, {}))
    totals = CellTotals("main", ref["genes"])
    totals.update(np.asarray(["a"]), np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="Missing receiver"):
        totals.means()


def test_reference_annotations_and_scaler_are_frozen():
    clean, genes = profiles()
    scaler, labels, encoded, indices, _ = annotation_protocol(clean, genes)
    assert set(labels) == set(CLASS_ORDER)
    mean = scaler.mean_.copy()
    record = annotation_metric_record(clean, scaler=scaler, reference_labels=labels,
                                      encoded_reference=encoded, marker_indices=indices)
    for metric in ("accuracy", "macro_precision", "balanced_accuracy", "macro_f1", "mcc", "cohen_kappa"):
        assert record[metric] == 1
    assert np.asarray(record["confusion_matrix"]).trace() == len(clean)
    annotation_metric_record(clean[::-1], scaler=scaler, reference_labels=labels,
                             encoded_reference=encoded, marker_indices=indices)
    np.testing.assert_array_equal(scaler.mean_, mean)


def test_missing_published_marker_cannot_silently_reduce_program():
    clean, genes = profiles()
    with pytest.raises(ValueError, match="lacks published markers"):
        annotation_protocol(clean[:, 1:], genes[1:])


def test_clustering_repeats_fixed_initializations_and_uses_frozen_labels():
    clean, _ = profiles()
    labels = np.repeat(np.arange(5), 16).astype(str)
    latent = independent_pca(clean)
    assert latent.shape == (80, 20)
    scores = clustering_metrics(latent, labels)
    assert scores == clustering_metrics(latent, labels)
    shuffled = clustering_metrics(latent, np.roll(labels, 8))
    assert scores["source_agreement_ari"] > shuffled["source_agreement_ari"]
    assert scores["frozen_label_silhouette_20pc"] > shuffled["frozen_label_silhouette_20pc"]


def test_structure_fold_mean_precedes_training_seed_sd():
    results = []
    for seed, values in ((1, (0, 3, 6)), (2, (6, 9, 12)), (3, (12, 15, 18))):
        for split, value in zip(("a9", "l7", "na"), values):
            results.append({"split": split, "seed": seed, **dict.fromkeys(METRICS, value),
                            "structure": {"states": {state: dict.fromkeys(STRUCTURE_METRICS, value if state == "prediction" else 1)
                                                      for state in ("reference", "mixed", "prediction")}}})
    scores = summarize_results(results)["loco"]["structure"]
    assert scores["prediction"]["macro_f1"] == {"mean": 9., "sample_sd": 6.}
    assert scores["mixed"]["macro_f1"] == {"mean": 1., "sample_sd": None}
    assert "macro_average_precision" not in scores["reference"]


def test_partial_structure_summary_is_rejected():
    with pytest.raises(ValueError, match="cover every run"):
        summarize_results([{"structure": {}}, {}])


def test_structure_is_available_for_all_published_plans():
    plan = run("all", [20260816], structure=True, dry_run=True)
    assert len(plan) == 7
    assert all(item["commands"][-1][-1] == "--structure" for item in plan)


@pytest.mark.parametrize("split,cells", [("main", 6394), ("a9", 2027), ("l7", 2127), ("na", 2240)])
def test_frozen_cohort_resource_and_gene_order(split, cells):
    from phygrec.structure_evaluation import RESOURCES
    with np.load(RESOURCES / f"{split}.npz", allow_pickle=False) as bank:
        genes = bank["genes"]
    reference, metadata = load_reference(split, genes)
    assert len(reference["ids"]) == metadata["cells"] == cells
    assert len(set(reference["ids"])) == cells
    with pytest.raises(ValueError, match="gene panel"):
        load_reference(split, genes[::-1])
