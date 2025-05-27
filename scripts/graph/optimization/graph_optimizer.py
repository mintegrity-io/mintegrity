from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from scripts.commons.logging_config import get_logger
from scripts.graph.model.transactions_graph import TransactionsGraph, Node, NodeType, Edge
from scripts.graph.categorization.wallet_categorizer import categorize_wallet, WalletType

log = get_logger()


class GraphOptimizer:
    """
    A class to optimize transaction graphs by removing dust wallets, inactive wallets,
    and other nodes that contribute to graph complexity without adding significant value.
    """

    def __init__(self,
                 remove_dust_wallets: bool = True,
                 remove_inactive_wallets: bool = True,
                 dust_wallet_tx_threshold: int = 5,
                 dust_wallet_value_threshold: float = 100,
                 inactive_days_threshold: int = 60,
                 min_transaction_value: float = 0.1):
        """
        Initialize the graph optimizer with configurable thresholds.

        Args:
            remove_dust_wallets: Whether to remove dust wallets
            remove_inactive_wallets: Whether to remove inactive wallets
            dust_wallet_tx_threshold: Maximum number of transactions for a dust wallet
            dust_wallet_value_threshold: Maximum USD value for a dust wallet
            inactive_days_threshold: Number of days without activity to consider a wallet inactive
            min_transaction_value: Minimum USD value for a transaction to be included
        """
        self.remove_dust_wallets = remove_dust_wallets
        self.remove_inactive_wallets = remove_inactive_wallets
        self.dust_wallet_tx_threshold = dust_wallet_tx_threshold
        self.dust_wallet_value_threshold = dust_wallet_value_threshold
        self.inactive_days_threshold = inactive_days_threshold
        self.min_transaction_value = min_transaction_value

    def optimize_graph(self, graph: TransactionsGraph) -> TransactionsGraph:
        """
        Create an optimized version of the input graph by removing dust wallets,
        inactive wallets, and low-value transactions.

        Args:
            graph: The original transaction graph

        Returns:
            A new optimized TransactionsGraph
        """
        log.info(f"Optimizing graph with {graph.get_number_of_nodes()} nodes and {graph.get_number_of_transactions()} transactions")

        # Create a new graph
        optimized_graph = TransactionsGraph()

        # Identify nodes to remove
        nodes_to_remove = self._identify_nodes_to_remove(graph)
        log.info(f"Identified {len(nodes_to_remove)} nodes to remove")

        # Copy nodes and edges, excluding the ones to remove
        self._copy_filtered_graph(graph, optimized_graph, nodes_to_remove)

        log.info(f"Optimized graph has {optimized_graph.get_number_of_nodes()} nodes and {optimized_graph.get_number_of_transactions()} transactions")
        return optimized_graph

    def _identify_nodes_to_remove(self, graph: TransactionsGraph) -> Set[str]:
        """
        Identify nodes that should be removed from the graph based on the configured criteria.

        Args:
            graph: The transaction graph

        Returns:
            A set of node addresses to remove
        """
        nodes_to_remove = set()
        current_time = datetime.now(timezone.utc)

        for address, node in graph.nodes.items():
            # Only consider wallet nodes for removal
            if node.type != NodeType.WALLET:
                continue

            # Use the wallet categorizer to determine the wallet type
            try:
                wallet_type = categorize_wallet(graph, node)

                # Check if it's a dust wallet
                if self.remove_dust_wallets and wallet_type == WalletType.DUST_WALLET:
                    nodes_to_remove.add(address)
                    log.debug(f"Removing dust wallet: {address}")
                    continue

                # Check if it's an inactive wallet
                if self.remove_inactive_wallets and wallet_type == WalletType.INACTIVE_WALLET:
                    nodes_to_remove.add(address)
                    log.debug(f"Removing inactive wallet: {address}")
                    continue

            except Exception as e:
                log.warning(f"Error categorizing wallet {address}: {str(e)}")

        return nodes_to_remove

    def _copy_filtered_graph(self,
                             source_graph: TransactionsGraph,
                             target_graph: TransactionsGraph,
                             nodes_to_remove: Set[str]):
        """
        Copy nodes and edges from the source graph to the target graph,
        excluding nodes in the nodes_to_remove set and their associated edges.

        Args:
            source_graph: The original graph
            target_graph: The new graph to populate
            nodes_to_remove: Set of node addresses to exclude
        """
        # First, copy all nodes except those in nodes_to_remove
        for address, node in source_graph.nodes.items():
            if address not in nodes_to_remove:
                target_graph.add_node_if_not_exists(node.address, node.type == NodeType.ROOT)

        # Then copy all edges where neither endpoint is in nodes_to_remove
        for edge_key, edge in source_graph.edges.items():
            from_addr = edge.from_node.address.address
            to_addr = edge.to_node.address.address

            if from_addr not in nodes_to_remove and to_addr not in nodes_to_remove:
                # Get the nodes in the target graph
                from_node = target_graph.get_node_by_address(edge.from_node.address)
                to_node = target_graph.get_node_by_address(edge.to_node.address)

                if from_node and to_node:
                    # Copy transactions with value above threshold
                    for tx_hash, tx in edge.transactions.items():
                        if tx.value_usd >= self.min_transaction_value:
                            target_graph.add_transaction(tx)


def optimize_transactions_graph(graph: TransactionsGraph,
                                remove_dust_wallets: bool = True,
                                remove_inactive_wallets: bool = True,
                                dust_wallet_tx_threshold: int = 5,
                                dust_wallet_value_threshold: float = 100,
                                inactive_days_threshold: int = 60,
                                min_transaction_value: float = 0.1) -> TransactionsGraph:
    """
    Convenience function to optimize a transaction graph by removing dust wallets,
    inactive wallets, and low-value transactions.

    Args:
        graph: The original transaction graph
        remove_dust_wallets: Whether to remove dust wallets
        remove_inactive_wallets: Whether to remove inactive wallets
        dust_wallet_tx_threshold: Maximum number of transactions for a dust wallet
        dust_wallet_value_threshold: Maximum USD value for a dust wallet
        inactive_days_threshold: Number of days without activity to consider a wallet inactive
        min_transaction_value: Minimum USD value for a transaction to be included

    Returns:
        A new optimized TransactionsGraph
    """
    optimizer = GraphOptimizer(
        remove_dust_wallets=remove_dust_wallets,
        remove_inactive_wallets=remove_inactive_wallets,
        dust_wallet_tx_threshold=dust_wallet_tx_threshold,
        dust_wallet_value_threshold=dust_wallet_value_threshold,
        inactive_days_threshold=inactive_days_threshold,
        min_transaction_value=min_transaction_value
    )

    return optimizer.optimize_graph(graph)
