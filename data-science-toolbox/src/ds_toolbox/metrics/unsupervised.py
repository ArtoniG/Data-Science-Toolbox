"""
src/credit_toolbox/metrics/unsupervised.py

Core statistical metrics for Unsupervised Learning (Clustering and Segmentation).
Used primarily in credit risk for behavioral profiling, vintage analysis segmentation, 
and sub-population discovery. Includes standard metrics and manual implementations 
for indices not natively supported by Scikit-Learn.
"""

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score

from credit_toolbox.core.exceptions import MetricCalculationError
from credit_toolbox.core.types import ArrayLike
from credit_toolbox.logging.decorators import log_execution_time


@log_execution_time
def calculate_silhouette(X: ArrayLike, labels: ArrayLike, sample_size: int | None = 10000) -> float:
    """
    Calculates the Silhouette Score to evaluate cluster density and separation.
    Values range from -1 (incorrect clustering) to +1 (highly dense, well-separated).
    
    Args:
        X: The feature matrix.
        labels: The cluster labels assigned to each row in X.
        sample_size: Downsamples the data for calculation to prevent memory overflow (O(N^2) complexity).
                     Defaults to 10,000.
    """
    try:
        # Silhouette score computes pairwise distances, which will cause Out-Of-Memory (OOM) 
        # errors on large credit bureau datasets if not sampled.
        return float(silhouette_score(X, labels, sample_size=sample_size))
    except ValueError as e:
        # Catches cases where there is only 1 cluster, or the number of labels 
        # equals the number of samples.
        raise MetricCalculationError(f"Failed to calculate Silhouette Score: {str(e)}")


@log_execution_time
def calculate_dunn_index(X: ArrayLike, labels: ArrayLike) -> float:
    """
    Calculates the Dunn Index.
    Defined as the ratio of the smallest inter-cluster distance to the largest intra-cluster distance.
    Higher values indicate better clustering (clusters are compact and well-separated).
    
    Since Scikit-Learn does not natively support the Dunn Index, this is a custom 
    vectorized implementation using Euclidean distance.
    """
    X_arr = np.array(X)
    labels_arr = np.array(labels)
    unique_labels = np.unique(labels_arr)

    if len(unique_labels) < 2:
        raise MetricCalculationError("Dunn Index requires at least 2 distinct clusters.")
    
    if len(unique_labels) == len(X_arr):
        raise MetricCalculationError("Dunn Index cannot be calculated when every point is its own cluster.")

    # Calculate the full distance matrix
    # Note: For massive datasets, this requires significant RAM. In production, 
    # it's recommended to pass a stratified sample of X to this function.
    try:
        distances = pairwise_distances(X_arr)
    except Exception as e:
        raise MetricCalculationError(f"Failed to compute pairwise distances: {str(e)}")

    max_intra_cluster_dist = 0.0
    min_inter_cluster_dist = float('inf')

    # Iterate through unique clusters to find diameters (intra) and separations (inter)
    for i, label_a in enumerate(unique_labels):
        
        # 1. Intra-cluster distance (Maximum distance between points in the SAME cluster)
        cluster_a_idx = np.where(labels_arr == label_a)[0]
        if len(cluster_a_idx) > 1:
            # np.ix_ extracts the sub-matrix of distances between points in cluster A
            intra_dist = np.max(distances[np.ix_(cluster_a_idx, cluster_a_idx)])
            max_intra_cluster_dist = max(max_intra_cluster_dist, intra_dist)
            
        # 2. Inter-cluster distance (Minimum distance between points in DIFFERENT clusters)
        for label_b in unique_labels[i + 1:]:
            cluster_b_idx = np.where(labels_arr == label_b)[0]
            # Sub-matrix of distances between points in cluster A and cluster B
            inter_dist = np.min(distances[np.ix_(cluster_a_idx, cluster_b_idx)])
            min_inter_cluster_dist = min(min_inter_cluster_dist, inter_dist)

    # Handle edge case where all points in a cluster are identical (distance = 0)
    if max_intra_cluster_dist == 0:
        return float('inf')

    return float(min_inter_cluster_dist / max_intra_cluster_dist)