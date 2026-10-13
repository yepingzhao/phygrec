"""Independent PCA and Leiden clustering against frozen Reference labels."""

from itertools import combinations
import warnings

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from phygrec import SEEDS
from phygrec.utils import normalize_log1p


def independent_pca(expression: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(normalize_log1p(expression)).astype(
        np.float32, copy=False
    )
    np.clip(scaled, -10.0, 10.0, out=scaled)
    return PCA(n_components=20, svd_solver="arpack", random_state=SEEDS[0]).fit_transform(
        scaled
    ).astype(np.float32, copy=False)


def mean_pairwise_ari(label_sets: list[np.ndarray]) -> float:
    values = [
        adjusted_rand_score(left, right)
        for left, right in combinations(label_sets, 2)
    ]
    return float(np.mean(values)) if values else 1.0


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def clustering_metrics(
    latent: np.ndarray,
    frozen_labels: np.ndarray,
) -> dict[str, float]:
    silhouette = silhouette_score(
        latent,
        frozen_labels,
        metric="euclidean",
        sample_size=min(4000, len(latent)),
        random_state=SEEDS[0],
    )
    graph = ad.AnnData(np.zeros((len(latent), 1), dtype=np.float32))
    graph.obsm["X_state_pca"] = latent
    sc.pp.neighbors(
        graph,
        n_neighbors=30,
        use_rep="X_state_pca",
        metric="cosine",
        random_state=SEEDS[0],
    )
    label_sets: list[np.ndarray] = []
    agreement_ari: list[float] = []
    agreement_nmi: list[float] = []
    cluster_counts: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in SEEDS:
            key = f"leiden_{seed}"
            sc.tl.leiden(
                graph,
                resolution=0.35,
                flavor="leidenalg",
                random_state=seed,
                key_added=key,
            )
            labels = graph.obs[key].astype(str).to_numpy()
            label_sets.append(labels)
            agreement_ari.append(adjusted_rand_score(frozen_labels, labels))
            agreement_nmi.append(normalized_mutual_info_score(frozen_labels, labels))
            cluster_counts.append(float(len(np.unique(labels))))
    ari_mean, ari_sd = mean_sd(agreement_ari)
    nmi_mean, nmi_sd = mean_sd(agreement_nmi)
    count_mean, count_sd = mean_sd(cluster_counts)
    return {
        "frozen_label_silhouette_20pc": float(silhouette),
        "source_agreement_ari": ari_mean,
        "source_agreement_ari_leiden_sd": ari_sd,
        "source_agreement_nmi": nmi_mean,
        "source_agreement_nmi_leiden_sd": nmi_sd,
        "leiden_stability_ari": mean_pairwise_ari(label_sets),
        "leiden_clusters": count_mean,
        "leiden_clusters_leiden_sd": count_sd,
    }
