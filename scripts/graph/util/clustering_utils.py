import json
import numpy as np

def clustering_result_to_json(result, feature_indices=None):
    """
    Convert a ClusteringResult object to a JSON-serializable dictionary

    Args:
        result: ClusteringResult object
        feature_indices: Optional tuple of feature indices used for visualization

    Returns:
        Dictionary with cluster data that can be serialized to JSON
    """
    # Convert numpy arrays and other non-serializable types to standard Python types

    # Convert cluster_stats keys from numpy.int64 to regular Python int
    # and ensure all values are JSON serializable
    converted_stats = {}
    for cluster_id, stats in result.cluster_stats.items():
        # Convert numpy.int64 to int for the key
        cluster_id_int = int(cluster_id)

        # Convert any numpy types in the stats values to Python native types
        converted_stats_values = {}
        for key, value in stats.items():
            # Convert numpy types to Python native types
            if hasattr(value, 'dtype') and hasattr(value, 'tolist'):
                # Handle numpy arrays
                converted_stats_values[key] = value.tolist()
            elif hasattr(value, 'dtype') and hasattr(value, 'item'):
                # Handle numpy scalars
                converted_stats_values[key] = value.item()
            else:
                # Regular Python types
                converted_stats_values[key] = value

        converted_stats[cluster_id_int] = converted_stats_values

    # Convert cluster assignments values to ensure they are regular Python types
    converted_assignments_values = [int(cluster_id) for cluster_id in result.cluster_assignments.values()]

    # Convert feature_indices to regular Python types if it's not None
    converted_feature_indices = None
    if feature_indices is not None:
        converted_feature_indices = tuple(int(idx) for idx in feature_indices)

    json_data = {
        "cluster_stats": converted_stats,
        "cluster_count": len(set(converted_assignments_values)) - (1 if -1 in converted_assignments_values else 0),
        "noise_count": converted_assignments_values.count(-1) if -1 in converted_assignments_values else 0,
        "total_wallets": len(result.cluster_assignments),
        "feature_indices": converted_feature_indices
    }

    # Add cluster centers if available (convert from numpy array)
    if result.cluster_centers is not None:
        json_data["cluster_centers"] = result.cluster_centers.tolist()

    # Add cluster assignments (limited to 1000 addresses to keep file size reasonable)
    sample_assignments = {}
    for i, (addr, cluster_id) in enumerate(result.cluster_assignments.items()):
        if i >= 1000:  # Limit to 1000 addresses
            break
        sample_assignments[addr] = int(cluster_id)

    json_data["sample_assignments"] = sample_assignments

    return json_data