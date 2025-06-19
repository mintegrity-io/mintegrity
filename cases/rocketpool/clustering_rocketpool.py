import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.commons import prices
from scripts.graph.analysis.clustering.wallet_clustering import extract_wallet_features, cluster_wallets_kmeans, visualize_clusters_2d, cluster_wallets_dbscan
from scripts.graph.categorization.graph_categorizer import CategorizedNode, categorize_graph
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.util.clustering_utils import clustering_result_to_json

prices.init()


GRAPH_NAME = "rocket_pool_full_graph_90_days"
GRAPH_PATH = f"./files/{GRAPH_NAME}.json"
graph = TransactionsGraph.from_dict(json.load(open(GRAPH_PATH)))

categorized_nodes: dict[str, CategorizedNode] = categorize_graph(graph)

# Create output directory if it doesn't exist
output_dir = Path("./files/clustering")
output_dir.mkdir(exist_ok=True, parents=True)

# Extract features from wallet nodes
print("Extracting wallet features...")
wallet_features = extract_wallet_features(graph, categorized_nodes)
print(f"Extracted features for {len(wallet_features)} wallets")

# Apply KMeans clustering with different numbers of clusters
print("Applying KMeans clustering...")
for n_clusters in [3, 5, 7]:
    kmeans_result = cluster_wallets_kmeans(wallet_features, n_clusters=n_clusters)

    # Create visualization
    print(f"Creating KMeans visualization with {n_clusters} clusters...")

    # Total Value vs Max TX Value
    fig1 = visualize_clusters_2d(kmeans_result, feature_indices=(0, 2))
    png_path1 = output_dir / f"wallet_clusters_kmeans_{n_clusters}_value_vs_max.png"
    fig1.savefig(png_path1)
    plt.close(fig1)

    # Save corresponding JSON with cluster data
    json_path1 = png_path1.with_suffix('.json')
    with open(json_path1, 'w') as f:
        json.dump(clustering_result_to_json(kmeans_result, feature_indices=(0, 2)), f, indent=2)

    # Total TXs vs Counterparties
    fig2 = visualize_clusters_2d(kmeans_result, feature_indices=(3, 6))
    png_path2 = output_dir / f"wallet_clusters_kmeans_{n_clusters}_tx_vs_counterparties.png"
    fig2.savefig(png_path2)
    plt.close(fig2)

    # Save corresponding JSON with cluster data
    json_path2 = png_path2.with_suffix('.json')
    with open(json_path2, 'w') as f:
        json.dump(clustering_result_to_json(kmeans_result, feature_indices=(3, 6)), f, indent=2)

    # Contract Ratio vs Unique Contracts
    fig3 = visualize_clusters_2d(kmeans_result, feature_indices=(9, 10))
    png_path3 = output_dir / f"wallet_clusters_kmeans_{n_clusters}_contract_interactions.png"
    fig3.savefig(png_path3)
    plt.close(fig3)

    # Save corresponding JSON with cluster data
    json_path3 = png_path3.with_suffix('.json')
    with open(json_path3, 'w') as f:
        json.dump(clustering_result_to_json(kmeans_result, feature_indices=(9, 10)), f, indent=2)

    # Print cluster statistics
    print(f"\nKMeans Clustering Results ({n_clusters} clusters):")
    for cluster_id, stats in kmeans_result.cluster_stats.items():
        print(f"Cluster {cluster_id}:")
        print(f"  Size: {stats['size']} wallets")
        print(f"  Average Transaction Value: ${stats['avg_total_value_usd']:.2f}")
        print(f"  Average Transaction Count: {stats['avg_tx_count']:.1f}")
        print(f"  Average Contract Interaction Ratio: {stats['avg_contract_ratio']:.2f}")
        if 'dominant_category' in stats:
            print(f"  Dominant Category: {stats['dominant_category']} ({stats['category_purity']:.1%})")
        print()

# Apply DBSCAN clustering with different parameters
print("\nApplying DBSCAN clustering...")
for eps, min_samples in [(0.5, 5), (0.7, 3), (0.3, 10)]:
    dbscan_result = cluster_wallets_dbscan(wallet_features, eps=eps, min_samples=min_samples)

    # Create visualization
    print(f"Creating DBSCAN visualization with eps={eps}, min_samples={min_samples}...")

    # Total Value vs Max TX Value
    fig1 = visualize_clusters_2d(dbscan_result, feature_indices=(0, 2))
    png_path1 = output_dir / f"wallet_clusters_dbscan_eps{eps}_ms{min_samples}_value_vs_max.png"
    fig1.savefig(png_path1)
    plt.close(fig1)

    # Save corresponding JSON with cluster data
    json_path1 = png_path1.with_suffix('.json')
    with open(json_path1, 'w') as f:
        json.dump(clustering_result_to_json(dbscan_result, feature_indices=(0, 2)), f, indent=2)

    # Total TXs vs Counterparties
    fig2 = visualize_clusters_2d(dbscan_result, feature_indices=(3, 6))
    png_path2 = output_dir / f"wallet_clusters_dbscan_eps{eps}_ms{min_samples}_tx_vs_counterparties.png"
    fig2.savefig(png_path2)
    plt.close(fig2)

    # Save corresponding JSON with cluster data
    json_path2 = png_path2.with_suffix('.json')
    with open(json_path2, 'w') as f:
        json.dump(clustering_result_to_json(dbscan_result, feature_indices=(3, 6)), f, indent=2)

    # Count number of clusters and noise points
    n_clusters = len(set(dbscan_result.cluster_assignments.values())) - (1 if -1 in dbscan_result.cluster_assignments.values() else 0)
    n_noise = list(dbscan_result.cluster_assignments.values()).count(-1)

    print(f"\nDBSCAN Clustering Results (eps={eps}, min_samples={min_samples}):")
    print(f"Found {n_clusters} clusters with {n_noise} noise points")

    # Print cluster statistics
    for cluster_id, stats in dbscan_result.cluster_stats.items():
        if cluster_id == -1:
            print(f"Noise points:")
        else:
            print(f"Cluster {cluster_id}:")
        print(f"  Size: {stats['size']} wallets")
        print(f"  Average Transaction Value: ${stats['avg_total_value_usd']:.2f}")
        print(f"  Average Transaction Count: {stats['avg_tx_count']:.1f}")
        if 'dominant_category' in stats:
            print(f"  Dominant Category: {stats['dominant_category']} ({stats['category_purity']:.1%})")
        print()

print("\nClustering analysis complete!")
print(f"Results saved to: {output_dir}")
print("PNG files contain visualizations, and corresponding JSON files contain complete cluster data.")
print("Use the JSON files to access all cluster information, especially when there are many clusters that may be cropped from the plots.")

# Optional: Inspect some clusters in more detail
# Get a sample of addresses from the largest KMeans cluster
largest_cluster_id = max(kmeans_result.cluster_stats, key=lambda k: kmeans_result.cluster_stats[k]['size'])
sample_addresses = kmeans_result.get_addresses_in_cluster(largest_cluster_id)[:5]

print(f"\nSample addresses from largest cluster (Cluster {largest_cluster_id}):")
for addr in sample_addresses:
    wallet_type = categorized_nodes[addr].type if addr in categorized_nodes and categorized_nodes[addr].type else "Uncategorized"
    print(f"Address: {addr[:10]}... - Category: {wallet_type}")
