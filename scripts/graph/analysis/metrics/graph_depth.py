"""
Graph depth metrics for transaction graphs.

This module provides functions to calculate various depth metrics for transaction graphs,
including maximum depth and average depth from a root node.
"""
from collections import deque
from typing import Dict, Tuple, List, Optional, Set, DefaultDict
from collections import defaultdict

from scripts.graph.model.transactions_graph import TransactionsGraph, Node, NodeType
from scripts.commons.logging_config import get_logger

log = get_logger()

def calculate_graph_depth_metrics(graph: TransactionsGraph, root_address: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate depth metrics for a transaction graph from a specified root node.

    Args:
        graph: The transaction graph to analyze
        root_address: The address of the root node. If None, will look for a node of type ROOT.

    Returns:
        Dictionary containing metrics:
        - 'max_depth': Maximum depth from root
        - 'avg_depth': Average depth from root
        - 'total_nodes_reached': Number of nodes reachable from root
    """
    # Find the root node
    if root_address:
        # Use the provided address
        if root_address not in graph.nodes:
            raise ValueError(f"Root address {root_address} not found in graph")
        root_node = graph.nodes[root_address]
    else:
        # Look for a node with type ROOT
        root_nodes = [node for node in graph.nodes.values() if node.type == NodeType.ROOT]
        if not root_nodes:
            raise ValueError("No ROOT node found in graph and no root_address provided")
        if len(root_nodes) > 1:
            log.warning(f"Multiple ROOT nodes found in graph, using the first one: {root_nodes[0].address.address}")
        root_node = root_nodes[0]

    # BFS to calculate depths
    depths = {}  # address -> depth
    visited = set()
    queue = deque([(root_node.address.address, 0)])  # (node_address, depth)

    while queue:
        node_address, depth = queue.popleft()

        if node_address in visited:
            continue

        visited.add(node_address)
        depths[node_address] = depth

        # Find all outgoing edges from this node
        for edge_key, edge in graph.edges.items():
            if edge_key[0] == node_address and edge_key[1] not in visited:
                queue.append((edge_key[1], depth + 1))

    # Calculate metrics
    if not depths:
        return {
            'max_depth': 0,
            'avg_depth': 0,
            'total_nodes_reached': 0
        }

    max_depth = max(depths.values())
    avg_depth = sum(depths.values()) / len(depths) if depths else 0
    total_nodes_reached = len(depths)

    return {
        'max_depth': max_depth,
        'avg_depth': avg_depth,
        'total_nodes_reached': total_nodes_reached
    }

def print_depth_distribution(graph: TransactionsGraph, root_address: Optional[str] = None) -> None:
    """
    Print a distribution of nodes by depth from the root node.

    Args:
        graph: The transaction graph to analyze
        root_address: The address of the root node. If None, will look for a node of type ROOT.
    """
    # Find the root node
    if root_address:
        # Use the provided address
        if root_address not in graph.nodes:
            raise ValueError(f"Root address {root_address} not found in graph")
        root_node = graph.nodes[root_address]
    else:
        # Look for a node with type ROOT
        root_nodes = [node for node in graph.nodes.values() if node.type == NodeType.ROOT]
        if not root_nodes:
            raise ValueError("No ROOT node found in graph and no root_address provided")
        if len(root_nodes) > 1:
            log.warning(f"Multiple ROOT nodes found in graph, using the first one: {root_nodes[0].address.address}")
        root_node = root_nodes[0]

    # BFS to calculate depths
    depths = {}  # address -> depth
    visited = set()
    queue = deque([(root_node.address.address, 0)])  # (node_address, depth)

    while queue:
        node_address, depth = queue.popleft()

        if node_address in visited:
            continue

        visited.add(node_address)
        depths[node_address] = depth

        # Find all outgoing edges from this node
        for edge_key, edge in graph.edges.items():
            if edge_key[0] == node_address and edge_key[1] not in visited:
                queue.append((edge_key[1], depth + 1))

    # Create distribution of nodes by depth
    depth_distribution = {}
    for _, depth in depths.items():
        if depth not in depth_distribution:
            depth_distribution[depth] = 0
        depth_distribution[depth] += 1

    # Print distribution
    log.info(f"{'=' * 50}")
    log.info(f"DEPTH DISTRIBUTION FROM ROOT {root_node.address.address}")
    log.info(f"{'=' * 50}")

    for depth in sorted(depth_distribution.keys()):
        count = depth_distribution[depth]
        percentage = (count / len(graph.nodes)) * 100
        log.info(f"Depth {depth}: {count} nodes ({percentage:.2f}% of total)")

    # Calculate metrics
    max_depth = max(depths.values()) if depths else 0
    avg_depth = sum(depths.values()) / len(depths) if depths else 0
    reachable_percentage = (len(depths) / len(graph.nodes)) * 100

    log.info(f"{'=' * 50}")
    log.info(f"Maximum depth: {max_depth}")
    log.info(f"Average depth: {avg_depth:.2f}")
    log.info(f"Total nodes reached: {len(depths)} ({reachable_percentage:.2f}% of all nodes)")
    log.info(f"{'=' * 50}")


def calculate_distances_to_roots(graph: TransactionsGraph) -> Dict[str, Dict[str, int]]:
    """
    Calculate the distance from each node to each root node in the graph.

    Args:
        graph: The transaction graph to analyze

    Returns:
        Dictionary mapping node addresses to dictionaries of root addresses and distances
        {node_address: {root_address: distance, ...}, ...}
    """
    # Find all root nodes
    root_nodes = [node for node in graph.nodes.values() if node.type == NodeType.ROOT]
    if not root_nodes:
        raise ValueError("No ROOT nodes found in graph")

    # Initialize result dictionary
    node_to_roots_distances = defaultdict(dict)

    # For each root node, calculate distances to all other nodes
    for root_node in root_nodes:
        root_address = root_node.address.address

        # BFS to calculate depths
        depths = {}  # address -> depth
        visited = set()
        queue = deque([(root_address, 0)])  # (node_address, depth)

        while queue:
            node_address, depth = queue.popleft()

            if node_address in visited:
                continue

            visited.add(node_address)
            depths[node_address] = depth

            # Find all outgoing edges from this node
            for edge_key, edge in graph.edges.items():
                if edge_key[0] == node_address and edge_key[1] not in visited:
                    queue.append((edge_key[1], depth + 1))

        # Add distances to result dictionary
        for node_address, distance in depths.items():
            node_to_roots_distances[node_address][root_address] = distance

    return dict(node_to_roots_distances)


def count_nodes_by_distance_from_roots(graph: TransactionsGraph) -> Dict[str, Dict[int, int]]:
    """
    Count the number of nodes at each distance from each root node.

    Args:
        graph: The transaction graph to analyze

    Returns:
        Dictionary mapping root addresses to dictionaries of distances and node counts
        {root_address: {distance: count, ...}, ...}
    """
    # Calculate distances from each node to each root
    node_to_roots_distances = calculate_distances_to_roots(graph)

    # Initialize result dictionary
    root_to_distance_counts = defaultdict(lambda: defaultdict(int))

    # Count nodes at each distance for each root
    for node_distances in node_to_roots_distances.values():
        for root_address, distance in node_distances.items():
            root_to_distance_counts[root_address][distance] += 1

    # Convert defaultdicts to regular dicts for return
    return {
        root_address: dict(distance_counts)
        for root_address, distance_counts in root_to_distance_counts.items()
    }


def print_distances_to_roots(graph: TransactionsGraph) -> None:
    """
    Print the distances from each node to each root node in the graph.

    Args:
        graph: The transaction graph to analyze
    """
    # Calculate distances from each node to each root
    node_to_roots_distances = calculate_distances_to_roots(graph)

    # Find all root nodes
    root_nodes = [node for node in graph.nodes.values() if node.type == NodeType.ROOT]

    log.info(f"{'=' * 50}")
    log.info(f"DISTANCES FROM NODES TO ROOT NODES")
    log.info(f"{'=' * 50}")

    # Print header with root addresses
    header = "Node Address"
    for root_node in root_nodes:
        root_address = root_node.address.address
        header += f" | Distance to {root_address}"
    log.info(header)
    log.info("-" * len(header))

    # Print distances for each node
    for node_address, root_distances in sorted(node_to_roots_distances.items()):
        line = f"{node_address}"
        for root_node in root_nodes:
            root_address = root_node.address.address
            distance = root_distances.get(root_address, "N/A")
            line += f" | {distance}"
        log.info(line)

    log.info(f"{'=' * 50}")


def print_nodes_count_by_distance(graph: TransactionsGraph) -> None:
    """
    Print the number of nodes at each distance from each root node.

    Args:
        graph: The transaction graph to analyze
    """
    # Count nodes at each distance from each root
    root_to_distance_counts = count_nodes_by_distance_from_roots(graph)

    log.info(f"{'=' * 50}")
    log.info(f"NODE COUNT BY DISTANCE FROM ROOT NODES")
    log.info(f"{'=' * 50}")

    # Print counts for each root
    for root_address, distance_counts in sorted(root_to_distance_counts.items()):
        log.info(f"Root: {root_address}")
        log.info("-" * 30)

        total_nodes = sum(distance_counts.values())

        for distance in sorted(distance_counts.keys()):
            count = distance_counts[distance]
            percentage = (count / total_nodes) * 100
            log.info(f"Distance {distance}: {count} nodes ({percentage:.2f}% of reachable nodes)")

        log.info(f"Total nodes reached from this root: {total_nodes}")
        log.info("")

    log.info(f"{'=' * 50}")
