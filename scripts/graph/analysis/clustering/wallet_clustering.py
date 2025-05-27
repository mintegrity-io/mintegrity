from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from scripts.commons.logging_config import get_logger
from scripts.graph.categorization.graph_categorizer import CategorizedNode
from scripts.graph.categorization.wallet_categorizer import WalletStats
from scripts.graph.model.transactions_graph import TransactionsGraph, NodeType

log = get_logger()


@dataclass
class WalletFeatures:
    """Features extracted from wallet data for clustering"""
    address: str
    # Transaction volume features
    total_value_usd: float = 0
    avg_tx_value_usd: float = 0
    max_tx_value_usd: float = 0
    # Transaction count features
    total_tx_count: int = 0
    tx_count_in: int = 0
    tx_count_out: int = 0
    # Network features
    unique_counterparties: int = 0
    in_out_ratio: float = 1.0  # Default to balanced
    # Temporal features
    activity_days: float = 0
    # Contract interaction features
    contract_interaction_ratio: float = 0
    unique_contracts: int = 0
    # Optional categorization
    category: Optional[str] = None

    def to_feature_vector(self) -> List[float]:
        """Convert features to a vector for clustering algorithms"""
        # Cap extremely large values to prevent infinity or values too large for float64
        MAX_VALUE = 1e9  # Reasonable upper limit for financial values

        # Apply safe log transform that handles zeros and prevents infinity
        def safe_log1p(x):
            if x > MAX_VALUE:
                return np.log1p(MAX_VALUE)
            elif x < 0:
                return 0  # Handle negative values if they somehow occur
            else:
                return np.log1p(x)

        # Handle in_out_ratio specially to avoid infinity
        safe_ratio = min(1000, self.in_out_ratio) if self.in_out_ratio > 0 else 0

        return [
            safe_log1p(self.total_value_usd),  # Log transform for better distribution
            safe_log1p(self.avg_tx_value_usd),
            safe_log1p(self.max_tx_value_usd),
            safe_log1p(self.total_tx_count),
            safe_log1p(self.tx_count_in),
            safe_log1p(self.tx_count_out),
            safe_log1p(self.unique_counterparties),
            safe_log1p(safe_ratio),
            safe_log1p(self.activity_days),
            min(1.0, self.contract_interaction_ratio),  # Ensure ratio is between 0 and 1
            safe_log1p(self.unique_contracts)
        ]


@dataclass
class ClusteringResult:
    """Result of clustering analysis"""
    features: Dict[str, WalletFeatures] = field(default_factory=dict)
    cluster_assignments: Dict[str, int] = field(default_factory=dict)
    cluster_centers: Optional[np.ndarray] = None
    cluster_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def get_cluster_for_address(self, address: str) -> int:
        """Get the cluster ID for a given address"""
        return self.cluster_assignments.get(address, -1)  # -1 indicates no cluster

    def get_addresses_in_cluster(self, cluster_id: int) -> List[str]:
        """Get all addresses in a specific cluster"""
        return [addr for addr, cluster in self.cluster_assignments.items() if cluster == cluster_id]

    def get_cluster_summary(self, cluster_id: int) -> Dict[str, float]:
        """Get summary statistics for a cluster"""
        return self.cluster_stats.get(cluster_id, {})


def extract_wallet_features(graph: TransactionsGraph, categorized_nodes: dict[str, CategorizedNode]) -> Dict[str, WalletFeatures]:
    """
    Extract features from wallet nodes in the graph for clustering

    Args:
        graph: The transaction graph containing wallet nodes
        categorized_nodes: Optional dictionary of categorized nodes to include categories

    Returns:
        Dictionary mapping wallet addresses to their extracted features
    """
    wallet_features = {}

    # Process all wallet nodes
    for address, node in graph.nodes.items():
        if node.type != NodeType.WALLET:
            continue

        # Initialize wallet stats to extract features
        stats = WalletStats()
        contract_interactions = 0
        total_tx = 0

        # Collect transaction statistics (similar to wallet_categorizer)
        for (from_addr, to_addr), edge in graph.edges.items():
            if node.address.address == from_addr:
                tx_count = len(edge.transactions)
                stats.tx_count_out += tx_count
                total_tx += tx_count

                # Check if counterparty is a contract
                to_node = graph.nodes.get(to_addr)
                if to_node and to_node.type == NodeType.CONTRACT:
                    contract_interactions += tx_count
                    stats.contracts_interacted_with.add(to_addr)

                # Process outgoing transactions
                for tx in edge.transactions.values():
                    tx_value_usd = tx.value_usd
                    stats.total_usd_out += tx_value_usd
                    stats.tx_size_distribution.append(tx_value_usd)
                    stats.max_tx_value_usd = max(stats.max_tx_value_usd, tx_value_usd)

                    # Parse timestamp for temporal analysis
                    try:
                        tx_time = datetime.fromisoformat(tx.timestamp.replace('Z', '+00:00'))
                        if not stats.first_tx_timestamp or tx_time < stats.first_tx_timestamp:
                            stats.first_tx_timestamp = tx_time
                        if not stats.last_tx_timestamp or tx_time > stats.last_tx_timestamp:
                            stats.last_tx_timestamp = tx_time
                    except (ValueError, AttributeError):
                        log.warning(f"Could not parse timestamp {tx.timestamp}")

                    # Track counterparties
                    stats.unique_counterparties.add(to_addr)

            elif node.address.address == to_addr:
                tx_count = len(edge.transactions)
                stats.tx_count_in += tx_count
                total_tx += tx_count

                # Check if counterparty is a contract
                from_node = graph.nodes.get(from_addr)
                if from_node and from_node.type == NodeType.CONTRACT:
                    stats.contracts_interacted_with.add(from_addr)

                # Process incoming transactions
                for tx in edge.transactions.values():
                    tx_value_usd = tx.value_usd
                    stats.total_usd_in += tx_value_usd
                    stats.tx_size_distribution.append(tx_value_usd)
                    stats.max_tx_value_usd = max(stats.max_tx_value_usd, tx_value_usd)

                    # Parse timestamp for temporal analysis
                    try:
                        tx_time = datetime.fromisoformat(tx.timestamp.replace('Z', '+00:00'))
                        if not stats.first_tx_timestamp or tx_time < stats.first_tx_timestamp:
                            stats.first_tx_timestamp = tx_time
                        if not stats.last_tx_timestamp or tx_time > stats.last_tx_timestamp:
                            stats.last_tx_timestamp = tx_time
                    except (ValueError, AttributeError):
                        log.warning(f"Could not parse timestamp {tx.timestamp}")

                    # Track counterparties
                    stats.unique_counterparties.add(from_addr)

        # Calculate derived metrics
        stats.finalize()
        total_value_usd = stats.total_usd_in + stats.total_usd_out
        counterparties = len(stats.unique_counterparties)
        activity_days = stats.get_activity_duration_days()
        unique_contracts = len(stats.contracts_interacted_with)

        # Create feature object
        wallet_feature = WalletFeatures(
            address=node.address.address,
            total_value_usd=total_value_usd,
            avg_tx_value_usd=stats.avg_tx_value_usd,
            max_tx_value_usd=stats.max_tx_value_usd,
            total_tx_count=total_tx,
            tx_count_in=stats.tx_count_in,
            tx_count_out=stats.tx_count_out,
            unique_counterparties=counterparties,
            in_out_ratio=stats.in_out_ratio,
            activity_days=activity_days,
            contract_interaction_ratio=contract_interactions / max(1, total_tx),
            unique_contracts=unique_contracts
        )

        # Add category if available
        if categorized_nodes and node.address.address in categorized_nodes:
            cat_node = categorized_nodes[node.address.address]
            if cat_node.type:
                wallet_feature.category = str(cat_node.type)

        wallet_features[node.address.address] = wallet_feature

    log.info(f"Extracted features for {len(wallet_features)} wallet nodes")
    return wallet_features


def cluster_wallets_kmeans(features: Dict[str, WalletFeatures], n_clusters: int = 5) -> ClusteringResult:
    """
    Cluster wallets using KMeans algorithm

    Args:
        features: Dictionary of wallet features
        n_clusters: Number of clusters to create

    Returns:
        ClusteringResult containing cluster assignments and statistics
    """
    if len(features) < n_clusters:
        log.warning(f"Not enough wallets ({len(features)}) for {n_clusters} clusters. Reducing cluster count.")
        n_clusters = max(2, len(features) // 2)

    # Convert features to matrix
    addresses = list(features.keys())
    feature_matrix = np.array([features[addr].to_feature_vector() for addr in addresses])

    # Scale features for better clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # Create result
    result = ClusteringResult()
    result.features = features
    result.cluster_centers = kmeans.cluster_centers_

    # Assign clusters to addresses
    for i, address in enumerate(addresses):
        result.cluster_assignments[address] = int(cluster_labels[i])

    # Calculate cluster statistics
    for cluster_id in range(n_clusters):
        cluster_addrs = result.get_addresses_in_cluster(cluster_id)
        if not cluster_addrs:
            continue

        # Calculate statistics for this cluster
        stats = {
            "size": len(cluster_addrs),
            "avg_total_value_usd": np.mean([features[addr].total_value_usd for addr in cluster_addrs]),
            "avg_tx_count": np.mean([features[addr].total_tx_count for addr in cluster_addrs]),
            "avg_counterparties": np.mean([features[addr].unique_counterparties for addr in cluster_addrs]),
            "avg_contract_ratio": np.mean([features[addr].contract_interaction_ratio for addr in cluster_addrs]),
        }

        # Add category distribution if available
        categories = [features[addr].category for addr in cluster_addrs if features[addr].category]
        if categories:
            category_counts = Counter(categories)
            dominant_category = category_counts.most_common(1)[0][0]
            stats["dominant_category"] = dominant_category
            stats["category_purity"] = category_counts[dominant_category] / len(categories)

        result.cluster_stats[cluster_id] = stats

    log.info(f"KMeans clustering completed. Created {n_clusters} clusters.")
    return result


def cluster_wallets_dbscan(features: Dict[str, WalletFeatures], eps: float = 0.5, min_samples: int = 5) -> ClusteringResult:
    """
    Cluster wallets using DBSCAN algorithm, which can detect clusters of arbitrary shape
    and automatically determine the number of clusters

    Args:
        features: Dictionary of wallet features
        eps: Maximum distance between samples for them to be considered as in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood for a point to be a core point

    Returns:
        ClusteringResult containing cluster assignments and statistics
    """
    # Convert features to matrix
    addresses = list(features.keys())
    feature_matrix = np.array([features[addr].to_feature_vector() for addr in addresses])

    # Scale features for better clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Perform clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(scaled_features)

    # Create result
    result = ClusteringResult()
    result.features = features

    # Assign clusters to addresses
    for i, address in enumerate(addresses):
        result.cluster_assignments[address] = int(cluster_labels[i])

    # Calculate cluster statistics
    unique_clusters = set(cluster_labels)
    for cluster_id in unique_clusters:
        cluster_addrs = result.get_addresses_in_cluster(cluster_id)
        if not cluster_addrs:
            continue

        # Calculate statistics for this cluster
        stats = {
            "size": len(cluster_addrs),
            "avg_total_value_usd": np.mean([features[addr].total_value_usd for addr in cluster_addrs]),
            "avg_tx_count": np.mean([features[addr].total_tx_count for addr in cluster_addrs]),
            "avg_counterparties": np.mean([features[addr].unique_counterparties for addr in cluster_addrs]),
            "avg_contract_ratio": np.mean([features[addr].contract_interaction_ratio for addr in cluster_addrs]),
        }

        # Add category distribution if available
        categories = [features[addr].category for addr in cluster_addrs if features[addr].category]
        if categories:
            category_counts = Counter(categories)
            dominant_category = category_counts.most_common(1)[0][0]
            stats["dominant_category"] = dominant_category
            stats["category_purity"] = category_counts[dominant_category] / len(categories)

        result.cluster_stats[cluster_id] = stats

    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)  # -1 is noise in DBSCAN
    log.info(f"DBSCAN clustering completed. Found {n_clusters} clusters with {sum(1 for l in cluster_labels if l == -1)} noise points.")
    return result


def visualize_clusters_2d(clustering_result: ClusteringResult, feature_indices: Tuple[int, int] = (0, 2), graph_name: str = None) -> plt.Figure:
    """
    Create a 2D visualization of wallet clusters using specified features

    Args:
        clustering_result: Result from clustering
        feature_indices: Tuple of two indices selecting which features to plot
        graph_name: Optional name of the graph to include in the title

    Returns:
        Matplotlib figure object
    """
    # Extract data
    addresses = list(clustering_result.features.keys())
    feature_matrix = np.array([clustering_result.features[addr].to_feature_vector() for addr in addresses])
    cluster_labels = np.array([clustering_result.get_cluster_for_address(addr) for addr in addresses])

    # Create a color map with distinctive colors and handle noise (-1) separately
    unique_clusters = sorted(set(cluster_labels))
    is_dbscan = -1 in unique_clusters  # Check if DBSCAN was used

    if is_dbscan:
        unique_clusters.remove(-1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
        color_map[-1] = (0.5, 0.5, 0.5, 1.0)  # Gray for noise
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each cluster
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        markers = 'o' if cluster_id != -1 else 'x'
        label = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
        ax.scatter(
            feature_matrix[mask, feature_indices[0]],
            feature_matrix[mask, feature_indices[1]],
            s=50, color=color_map[cluster_id], marker=markers,
            alpha=0.7, label=label
        )

    # Add features names
    feature_names = [
        "Log Total Value (USD)",
        "Log Avg TX Value (USD)",
        "Log Max TX Value (USD)",
        "Log Total TXs",
        "Log Incoming TXs",
        "Log Outgoing TXs",
        "Log Unique Counterparties",
        "Log In/Out Ratio",
        "Log Activity Days",
        "Contract Interaction Ratio",
        "Log Unique Contracts"
    ]

    # Set labels and title
    ax.set_xlabel(feature_names[feature_indices[0]], fontsize=14)
    ax.set_ylabel(feature_names[feature_indices[1]], fontsize=14)

    # Create title with graph name if provided
    title = "Wallet Clusters Visualization"
    if graph_name:
        title = f"{title} - {graph_name}"
    ax.set_title(title, fontsize=16)

    # Add legend
    ax.legend(fontsize=12)

    # Add cluster statistics as text annotations
    y_pos = 0.98
    x_pos = 1.02
    line_height = 0.03
    for cluster_id, stats in sorted(clustering_result.cluster_stats.items()):
        if cluster_id == -1:  # Skip noise for cleaner annotation
            continue

        cluster_text = f"Cluster {cluster_id} ({stats['size']} wallets):"
        ax.annotate(cluster_text, xy=(x_pos, y_pos), xycoords='axes fraction', fontsize=10, fontweight='bold')
        y_pos -= line_height

        # Add dominant category if available
        if "dominant_category" in stats:
            category_text = f"  Category: {stats['dominant_category']} ({stats['category_purity']:.1%})"
            ax.annotate(category_text, xy=(x_pos, y_pos), xycoords='axes fraction', fontsize=9)
            y_pos -= line_height

        # Add key metrics
        metrics = [
            f"  Avg Value: ${stats['avg_total_value_usd']:.2f}",
            f"  Avg TXs: {stats['avg_tx_count']:.1f}",
            f"  Avg Contract Ratio: {stats['avg_contract_ratio']:.2f}"
        ]

        for metric in metrics:
            ax.annotate(metric, xy=(x_pos, y_pos), xycoords='axes fraction', fontsize=9)
            y_pos -= line_height

        y_pos -= line_height/2  # Extra space between clusters

    # Add diagonal line and explanation for value_vs_max plots (feature indices 0 and 2)
    if feature_indices == (0, 2):
        # Get the axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Plot the diagonal line (max_tx_value = total_value)
        limit = max(x_max, y_max)
        ax.plot([0, limit], [0, limit], 'r--', linewidth=2, alpha=0.7, label="max_tx_value = total_value")

        # Add annotation explaining the diagonal line
        ax.annotate(
            "Diagonal Line: max_tx_value = total_value\n"
            "- Mathematical constraint: max transaction value cannot exceed total value\n"
            "- Points on the line: wallets with a single transaction or one dominant transaction\n"
            "- Points below the line: wallets with multiple transactions of similar values",
            xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
        )

        # Update legend to include the diagonal line
        ax.legend(fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Make room for annotations

    return fig
