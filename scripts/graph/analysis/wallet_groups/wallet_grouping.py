"""
Wallet coordination detection - identifies groups of wallets likely controlled by the same operator
based on transaction patterns and timing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from scripts.commons.logging_config import get_logger
from scripts.graph.analysis.clustering.wallet_clustering import extract_wallet_features
from scripts.graph.model.transactions_graph import TransactionsGraph, Node, NodeType
from scripts.graph.visualization.transaction_graph_visualization import visualize_graph

log = get_logger()


@dataclass
class WalletCoordinationMetrics:
    """Metrics that indicate coordination between wallets"""
    # Direct transfers
    direct_transfers: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    transfer_volume_usd: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # Timing patterns
    sequential_txs: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Contract interactions
    same_contract_interactions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Service pattern detection
    # Some wallets act as "layer wallets" (interact with many contracts)
    # while others act as "storage wallets" (mostly hold assets)
    is_layer_wallet: bool = False
    is_storage_wallet: bool = False
    connected_storage_wallets: Set[str] = field(default_factory=set)
    connected_layer_wallets: Set[str] = field(default_factory=set)

    # Operational patterns
    reuse_same_tokens: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    similar_tx_values: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Coordination score with each other wallet
    coordination_scores: Dict[str, float] = field(default_factory=dict)


def detect_wallet_coordination(graph: TransactionsGraph) -> Dict[str, Dict[str, float]]:
    """
    Detect coordination between wallets in the graph

    Args:
        graph: Transaction graph to analyze

    Returns:
        Dictionary mapping wallet addresses to their coordination metrics with other wallets
    """
    log.info("Detecting wallet coordination patterns...")

    global wallet_metrics  # Use the global wallet_metrics

    # Step 1: Extract wallet nodes only
    wallet_nodes = {addr: node for addr, node in graph.nodes.items()
                    if node.type == NodeType.WALLET}

    if len(wallet_nodes) < 2:
        log.warning("Not enough wallet nodes to detect coordination")
        return {}

    log.info(f"Analyzing coordination patterns among {len(wallet_nodes)} wallets")

    # Step 2: Initialize coordination metrics for each wallet
    wallet_metrics = {
        addr: WalletCoordinationMetrics() for addr in wallet_nodes
    }

    # Step 3: Analyze transaction patterns
    _analyze_direct_transfers(graph, wallet_metrics)
    _analyze_timing_patterns(graph, wallet_metrics)
    _analyze_contract_interactions(graph, wallet_metrics)
    _detect_layer_storage_patterns(graph, wallet_metrics)

    # Step 4: Calculate coordination scores between wallets
    coordination_scores = _calculate_coordination_scores(wallet_metrics)

    log.info("Wallet coordination detection complete")
    return coordination_scores


def _analyze_direct_transfers(graph: TransactionsGraph,
                              wallet_metrics: Dict[str, WalletCoordinationMetrics]):
    """Analyze direct transfers between wallets"""
    log.info("Analyzing direct transfers between wallets...")

    # Look for edges between wallet nodes
    for (from_addr, to_addr), edge in graph.edges.items():
        # Check if both are wallets
        if (from_addr in wallet_metrics and to_addr in wallet_metrics):
            # Count transfers and volume between these wallets
            transfer_count = len(edge.transactions)
            volume_usd = sum(tx.value_usd for tx in edge.transactions.values())

            # Update metrics for both wallets
            wallet_metrics[from_addr].direct_transfers[to_addr] += transfer_count
            wallet_metrics[from_addr].transfer_volume_usd[to_addr] += volume_usd

            # Receiving wallet also tracks this relationship
            wallet_metrics[to_addr].direct_transfers[from_addr] += transfer_count
            wallet_metrics[to_addr].transfer_volume_usd[from_addr] += volume_usd


def _analyze_timing_patterns(graph: TransactionsGraph,
                             wallet_metrics: Dict[str, WalletCoordinationMetrics]):
    """Analyze timing patterns that suggest coordination"""
    log.info("Analyzing transaction timing patterns...")

    # Extract all transactions with timestamps
    wallet_txs = {}
    for wallet_addr in wallet_metrics:
        wallet_txs[wallet_addr] = []

    # Collect transactions for each wallet
    for (from_addr, to_addr), edge in graph.edges.items():
        for tx in edge.transactions.values():
            if from_addr in wallet_metrics:
                try:
                    tx_time = datetime.fromisoformat(tx.timestamp.replace('Z', '+00:00'))
                    wallet_txs[from_addr].append((tx_time, tx.transaction_hash))
                except (ValueError, AttributeError):
                    continue

    # Sort transactions by timestamp
    for wallet_addr in wallet_txs:
        wallet_txs[wallet_addr].sort()

    # Look for sequential transaction patterns between wallets
    # (One wallet transacts right after another within short time window)
    time_window = timedelta(minutes=5)  # Configurable time window

    # For each pair of wallets
    wallet_addresses = list(wallet_metrics.keys())
    for i, wallet1 in enumerate(wallet_addresses):
        for wallet2 in wallet_addresses[i + 1:]:
            # Skip if either wallet has no transactions
            if not wallet_txs[wallet1] or not wallet_txs[wallet2]:
                continue

            # Count sequential transactions within time window
            sequential_count = 0
            for tx1_time, tx1_hash in wallet_txs[wallet1]:
                for tx2_time, tx2_hash in wallet_txs[wallet2]:
                    time_diff = abs((tx2_time - tx1_time).total_seconds())
                    if time_diff <= time_window.total_seconds():
                        sequential_count += 1

            # Only record if significant sequential activity
            if sequential_count >= 3:  # Threshold for significance
                wallet_metrics[wallet1].sequential_txs[wallet2] = sequential_count
                wallet_metrics[wallet2].sequential_txs[wallet1] = sequential_count


def _analyze_contract_interactions(graph: TransactionsGraph,
                                   wallet_metrics: Dict[str, WalletCoordinationMetrics]):
    """Analyze patterns of wallets interacting with the same contracts"""
    log.info("Analyzing shared contract interactions...")

    # Build map of contracts each wallet interacts with
    wallet_contracts = {addr: set() for addr in wallet_metrics}

    for (from_addr, to_addr), edge in graph.edges.items():
        # If sender is wallet and receiver is contract
        if from_addr in wallet_metrics and to_addr in graph.nodes:
            to_node = graph.nodes[to_addr]
            if to_node.type == NodeType.CONTRACT:
                wallet_contracts[from_addr].add(to_addr)

    # Compare contract interactions between wallets
    wallet_addresses = list(wallet_metrics.keys())
    for i, wallet1 in enumerate(wallet_addresses):
        for wallet2 in wallet_addresses[i + 1:]:
            # Calculate intersection of contracts both wallets interact with
            common_contracts = wallet_contracts[wallet1].intersection(wallet_contracts[wallet2])

            if len(common_contracts) >= 2:  # Threshold for significance
                wallet_metrics[wallet1].same_contract_interactions[wallet2] = len(common_contracts)
                wallet_metrics[wallet2].same_contract_interactions[wallet1] = len(common_contracts)

                # Also track token reuse patterns
                for contract_addr in common_contracts:
                    # Get token types from transactions to this contract
                    for edge_key, edge in graph.edges.items():
                        if edge_key[0] == wallet1 and edge_key[1] == contract_addr:
                            for tx in edge.transactions.values():
                                if tx.token_symbol:
                                    wallet_metrics[wallet1].reuse_same_tokens[wallet2].add(tx.token_symbol)

                        if edge_key[0] == wallet2 and edge_key[1] == contract_addr:
                            for tx in edge.transactions.values():
                                if tx.token_symbol:
                                    wallet_metrics[wallet2].reuse_same_tokens[wallet1].add(tx.token_symbol)


def _detect_layer_storage_patterns(graph: TransactionsGraph,
                                   wallet_metrics: Dict[str, WalletCoordinationMetrics]):
    """
    Detect layer+storage wallet patterns:
    - Layer wallets: interact with many contracts, conduct many transactions
    - Storage wallets: mostly receive and hold assets, fewer outgoing transactions
    """
    log.info("Detecting layer and storage wallet patterns...")

    # Calculate basic metrics for each wallet
    wallet_stats = {}
    for wallet_addr in wallet_metrics:
        contract_interactions = 0
        total_txs_out = 0
        total_txs_in = 0
        total_value_in = 0
        total_value_out = 0
        unique_contracts = set()

        # Count outgoing transactions
        for (from_addr, to_addr), edge in graph.edges.items():
            if from_addr == wallet_addr:
                total_txs_out += len(edge.transactions)
                total_value_out += sum(tx.value_usd for tx in edge.transactions.values())

                # Check if destination is a contract
                if to_addr in graph.nodes and graph.nodes[to_addr].type == NodeType.CONTRACT:
                    contract_interactions += len(edge.transactions)
                    unique_contracts.add(to_addr)

            # Count incoming transactions
            if to_addr == wallet_addr:
                total_txs_in += len(edge.transactions)
                total_value_in += sum(tx.value_usd for tx in edge.transactions.values())

        # Store stats
        wallet_stats[wallet_addr] = {
            "contract_interactions": contract_interactions,
            "unique_contracts": len(unique_contracts),
            "total_txs_out": total_txs_out,
            "total_txs_in": total_txs_in,
            "in_out_ratio": (total_txs_in / max(1, total_txs_out)),
            "value_in": total_value_in,
            "value_out": total_value_out,
            "value_in_out_ratio": (total_value_in / max(1, total_value_out))
        }

    # Identify layer and storage wallets based on behavior patterns
    for wallet_addr, stats in wallet_stats.items():
        # Layer wallets interact with many contracts and have balanced in/out ratio
        if (stats["unique_contracts"] >= 3 and
                stats["contract_interactions"] > 5 and
                stats["total_txs_out"] >= stats["total_txs_in"] * 0.5):
            wallet_metrics[wallet_addr].is_layer_wallet = True
            log.info(f"Identified layer wallet: {wallet_addr}")

        # Storage wallets receive more than they send out, fewer contract interactions
        if (stats["in_out_ratio"] > 1.2 and
                stats["value_in_out_ratio"] > 1.5 and
                stats["value_in"] > 1000):  # Minimum USD threshold
            wallet_metrics[wallet_addr].is_storage_wallet = True
            log.info(f"Identified storage wallet: {wallet_addr}")

    # Connect layer and storage wallets that have direct transfers
    for layer_addr, metrics in wallet_metrics.items():
        if metrics.is_layer_wallet:
            for storage_addr in wallet_metrics:
                if wallet_metrics[storage_addr].is_storage_wallet:
                    # Check if there are direct transfers between them
                    if (layer_addr in wallet_metrics[storage_addr].direct_transfers or
                            storage_addr in metrics.direct_transfers):
                        # Add connection
                        metrics.connected_storage_wallets.add(storage_addr)
                        wallet_metrics[storage_addr].connected_layer_wallets.add(layer_addr)


def _calculate_coordination_scores(wallet_metrics: Dict[str, WalletCoordinationMetrics]) -> Dict[str, Dict[str, float]]:
    """Calculate coordination scores between wallets based on observed patterns"""
    log.info("Calculating coordination scores between wallet pairs...")

    coordination_scores = {addr: {} for addr in wallet_metrics}

    # For each wallet pair
    wallet_addresses = list(wallet_metrics.keys())
    for i, wallet1 in enumerate(wallet_addresses):
        metrics1 = wallet_metrics[wallet1]

        for wallet2 in wallet_addresses[i + 1:]:
            metrics2 = wallet_metrics[wallet2]
            score = 0.0

            # Factor 1: Direct transfers (strongest signal)
            direct_transfers = metrics1.direct_transfers.get(wallet2, 0)
            if direct_transfers > 0:
                # Scale by frequency, with diminishing returns
                transfer_score = min(5.0, np.log1p(direct_transfers))
                # Scale by USD volume
                volume_score = min(5.0, np.log1p(metrics1.transfer_volume_usd.get(wallet2, 0) / 1000))
                score += transfer_score + volume_score

            # Factor 2: Sequential transactions
            seq_tx_count = metrics1.sequential_txs.get(wallet2, 0)
            if seq_tx_count > 0:
                score += min(4.0, 0.5 * np.log1p(seq_tx_count))

            # Factor 3: Same contract interactions
            common_contracts = metrics1.same_contract_interactions.get(wallet2, 0)
            if common_contracts > 0:
                score += min(3.0, np.log1p(common_contracts))

            # Factor 4: Layer-Storage relationship
            if (wallet2 in metrics1.connected_storage_wallets or
                    wallet1 in metrics2.connected_layer_wallets):
                score += 5.0  # Strong signal

            # Factor 5: Token reuse
            common_tokens = len(metrics1.reuse_same_tokens.get(wallet2, set()))
            if common_tokens > 0:
                score += min(2.0, 0.5 * np.log1p(common_tokens))

            # Normalize score to 0-10 range and save if significant
            normalized_score = min(10.0, score)
            if normalized_score >= 1.0:  # Only record significant coordination
                coordination_scores[wallet1][wallet2] = normalized_score
                coordination_scores[wallet2][wallet1] = normalized_score

                # Also update the metrics objects
                metrics1.coordination_scores[wallet2] = normalized_score
                metrics2.coordination_scores[wallet1] = normalized_score

    return coordination_scores


def identify_wallet_groups(coordination_scores: Dict[str, Dict[str, float]],
                           wallet_metrics: Dict[str, WalletCoordinationMetrics],
                           threshold: float = 5.0) -> List[Set[str]]:
    """
    Identify groups of wallets that are likely controlled by the same operator
    based on coordination scores

    Args:
        coordination_scores: Dictionary of coordination scores between wallets
        wallet_metrics: Dictionary of wallet coordination metrics
        threshold: Minimum coordination score to consider wallets related

    Returns:
        List of wallet groups (sets of addresses)
    """
    log.info(f"Identifying wallet groups with coordination threshold {threshold}...")

    # Create undirected graph where edges represent strong coordination
    wallet_connections = defaultdict(set)
    for wallet1, connections in coordination_scores.items():
        for wallet2, score in connections.items():
            if score >= threshold:
                wallet_connections[wallet1].add(wallet2)
                wallet_connections[wallet2].add(wallet1)

    # Find connected components (wallet groups)
    wallet_groups = []
    processed_wallets = set()

    # For each wallet
    for wallet in coordination_scores:
        if wallet in processed_wallets:
            continue

        # Start a new group
        group = set()
        to_process = [wallet]

        # BFS to find all connected wallets
        while to_process:
            current = to_process.pop(0)
            if current in processed_wallets:
                continue

            group.add(current)
            processed_wallets.add(current)

            # Add connected wallets to processing queue
            for connected in wallet_connections[current]:
                if connected not in processed_wallets:
                    to_process.append(connected)

        # Only add groups with at least 2 wallets
        if len(group) >= 2:
            wallet_groups.append(group)

    # Sort groups by size (largest first)
    wallet_groups.sort(key=len, reverse=True)

    log.info(f"Identified {len(wallet_groups)} wallet groups")
    for i, group in enumerate(wallet_groups):
        # Check if wallet exists in metrics before accessing properties
        layer_wallets = sum(1 for addr in group if addr in wallet_metrics and wallet_metrics[addr].is_layer_wallet)
        storage_wallets = sum(1 for addr in group if addr in wallet_metrics and wallet_metrics[addr].is_storage_wallet)
        log.info(f"Group {i + 1}: {len(group)} wallets ({layer_wallets} layer, {storage_wallets} storage)")

    return wallet_groups


def analyze_and_visualize_wallet_groups(graph: TransactionsGraph, output_path: str, threshold: float = 5.0):
    """
    Analyze wallet coordination and visualize the identified groups

    Args:
        graph: Transaction graph to analyze
        output_path: Path to save visualization output
        threshold: Minimum coordination score to consider wallets related (0-10 scale)
    """
    # Step 1: Detect wallet coordination
    coordination_scores = detect_wallet_coordination(graph)

    # Step 2: Identify wallet groups
    global wallet_metrics  # Use the global wallet_metrics populated by detect_wallet_coordination
    wallet_groups = identify_wallet_groups(coordination_scores, wallet_metrics, threshold=threshold)

    # Step 3: Create visualization with highlighted groups
    group_colors = {}
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Store the mapping of group numbers to colors for legend creation
    group_num_to_color = {}

    # Assign colors to wallet groups
    for i, group in enumerate(wallet_groups):
        color = color_palette[i % len(color_palette)]
        group_num = i + 1  # Group number (1-based)
        group_num_to_color[group_num] = color
        for wallet in group:
            group_colors[wallet] = (color, group_num)  # Now storing tuple of (color, group_num)

    # Extract wallet features for additional information
    wallet_features = extract_wallet_features(graph, None)

    # Prepare node information for visualization
    node_info = {}
    for addr, node in graph.nodes.items():
        if node.type == NodeType.WALLET:
            # See if it belongs to a group
            group_id = None
            for i, group in enumerate(wallet_groups):
                if addr in group:
                    group_id = i + 1
                    break

            # Determine wallet type
            wallet_type = None
            if addr in wallet_metrics and wallet_metrics[addr].is_layer_wallet:
                wallet_type = "Layer Wallet"
            elif addr in wallet_metrics and wallet_metrics[addr].is_storage_wallet:
                wallet_type = "Storage Wallet"
            else:
                wallet_type = "Regular Wallet"

            # Get basic stats
            value_usd = 0
            tx_count = 0
            if addr in wallet_features:
                value_usd = wallet_features[addr].total_value_usd
                tx_count = wallet_features[addr].total_tx_count

            node_info[addr] = {
                "group": f"Group {group_id}" if group_id else "Ungrouped",
                "type": wallet_type,
                "total_value_usd": value_usd,
                "tx_count": tx_count
            }

    # Generate visualization
    visualize_graph(graph, output_path, node_colors=group_colors, node_info=node_info,
                    title="Wallet Groups Analysis - Potential Single-Operator Clusters")

    log.info(f"Visualization saved to {output_path}")
    return wallet_groups, coordination_scores


# Dictionary to store metrics for all wallets
wallet_metrics = {}

if __name__ == "__main__":
    # This can be run as a standalone script
    from scripts.graph.util.transactions_graph_json import load_graph_from_json

    # Example usage:
    # graph_path = "D:/workspace/mintegrity/scripts/graph/files/rocket_pool_graph_60_days.json"
    # output_path = "D:/workspace/mintegrity/scripts/graph/files/clustering_results/wallet_coordination_groups.html"
    # graph = load_graph_from_json(graph_path)
    # analyze_and_visualize_wallet_groups(graph, output_path)
