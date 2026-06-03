import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from src import hierarchical_sampling as hs
from sklearn.cluster import AgglomerativeClustering
from hdbscan import HDBSCAN


def _sample_from_labels(labels, target_size, seed=42):
    np.random.seed(seed)
    unique_labels = np.unique(labels)
    cluster_sizes = np.array([np.sum(labels == c) for c in unique_labels])
    
    target_sizes = hs.find_subcluster_target_size(cluster_sizes, target_size, 1)
    
    selected = []
    for c_label, c_target in zip(unique_labels, target_sizes):
        cluster_indices = np.where(labels == c_label)[0]
        selected.append(np.random.choice(cluster_indices, c_target, replace=False))
        
    return np.concatenate(selected).astype(np.int64)

def kmeans_sampling(features, target_size, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(features)
    return _sample_from_labels(kmeans.labels_, target_size)


def dbscan_sampling(features, target_size, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = db.labels_

    noise_idx = np.where(labels == -1)[0]
    valid_idx = np.where(labels >= 0)[0]

    n_from_clusters = min(target_size, len(valid_idx))
    selected = valid_idx[_sample_from_labels(labels[valid_idx], n_from_clusters)]

    remainder = target_size - len(selected)
    if remainder > 0 and len(noise_idx) > 0:
        n_noise = min(remainder, len(noise_idx))
        selected_noise = np.random.choice(noise_idx, n_noise, replace=False)
        selected = np.concatenate([selected, selected_noise])

    return selected.astype(np.int64)

def agglomerative_sampling(features, target_size, n_clusters=10, linkage="ward"):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(features)
    return _sample_from_labels(labels, target_size)

def hdbscan_sampling(features, target_size, min_cluster_size=50, min_samples=5):
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = hdb.fit_predict(features)

    noise_idx = np.where(labels == -1)[0]
    valid_idx = np.where(labels >= 0)[0]

    n_from_clusters = min(target_size, len(valid_idx))
    selected = valid_idx[_sample_from_labels(labels[valid_idx], n_from_clusters)]

    remainder = target_size - len(selected)
    if remainder > 0 and len(noise_idx) > 0:
        n_noise = min(remainder, len(noise_idx))
        selected_noise = np.random.choice(noise_idx, n_noise, replace=False)
        selected = np.concatenate([selected, selected_noise])

    return selected.astype(np.int64)