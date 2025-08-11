from typing import Tuple
import numpy as np
def cluster_embeddings(embeddings, min_cluster_size: int = 8):
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
        probs = getattr(clusterer, "probabilities_", None)
        if probs is None:
            probs = np.ones(len(labels))
        return labels, probs
    except Exception:
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=0.7, min_samples=min_cluster_size, metric="euclidean")
        labels = model.fit_predict(embeddings)
        probs = np.ones(len(labels))
        return labels, probs
def to_umap(embeddings, n_neighbors: int = 15, min_dist: float = 0.1):
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=42)
    return reducer.fit_transform(embeddings)
