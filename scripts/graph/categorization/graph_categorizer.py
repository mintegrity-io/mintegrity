from dataclasses import dataclass

from scripts.commons.logging_config import get_logger
from scripts.graph.categorization.contract_categorizer import ContractType, categorize_contract
from scripts.graph.categorization.wallet_categorizer import WalletType, categorize_wallet
from scripts.graph.model.transactions_graph import TransactionsGraph, NodeType, Node

log = get_logger()


@dataclass
class CategorizedNode:
    """
    A node with its assigned type classification
    """
    node: Node
    type: WalletType | ContractType | None


def categorize_graph(graph: TransactionsGraph) -> dict[str, CategorizedNode]:
    """
    Categorize all nodes in a transactions graph

    Args:
        graph: The transactions graph to categorize

    Returns:
        A dictionary mapping node addresses to their categorized types
    """
    result: dict[str, CategorizedNode] = {}

    log.info(f"Starting graph categorization for {len(graph.nodes)} nodes")

    for address, node in graph.nodes.items():
        category = None
        # Determine classification based on node type
        if node.type == NodeType.ROOT:
            # Root nodes are not categorized
            category = None
        elif node.type == NodeType.CONTRACT:
            # Use contract categorizer
            category = categorize_contract(graph, node)
        elif node.type == NodeType.WALLET:
            # Use wallet categorizer
            category = categorize_wallet(graph, node)

        result[address] = CategorizedNode(node=node, type=category)

        log.info(f"Categorized node {address} of type {node.type} as {category}")

    log.info(f"Completed graph categorization. Categorized {len(result)} nodes")
    return result
