import json
import os

from scripts.commons.logging_config import get_logger
from scripts.graph.model.transactions_graph import TransactionsGraph

log = get_logger()


def save_graph_to_json(graph: TransactionsGraph, filename: str):
    """
    Save the TransactionsGraph to a JSON file using the to_dict methods.

    :param graph: TransactionsGraph object to save
    :param filename: Path where to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create serializable dictionary with graph data and metadata
    serializable_graph = graph.to_dict()

    # Write to file
    with open(filename, 'w') as f:
        json.dump(serializable_graph, f, indent=2)

    log.info(f"Graph saved to {filename}")


def load_graph_from_json(filename: str) -> TransactionsGraph:
    """
    Load a TransactionsGraph from a JSON file.

    :param filename: Path to the JSON file
    :return: Reconstructed TransactionsGraph
    """
    log.info(f"Loading graph from {filename}")

    with open(filename, 'r') as f:
        data = json.load(f)

    # Remove stats if present (not needed for reconstruction)
    if "stats" in data:
        data.pop("stats")

    graph = TransactionsGraph.from_dict(data)

    log.info(f"Loaded graph with {graph.get_number_of_nodes()} nodes and {graph.get_number_of_transactions()} transactions")
    return graph