import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from src import hierarchical_sampling as hs


def _sample_from_labels(labels, target_size):
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
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    return _sample_from_labels(dbscan.labels_, target_size)