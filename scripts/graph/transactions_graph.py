from dataclasses import dataclass
from enum import IntEnum

from scripts.commons.logging_config import get_logger
from scripts.commons.model import *

log = get_logger()


class NodeType(IntEnum):
    # TODO I want to store the type of address in the address object and not in the node
    WALLET = 0
    CONTRACT = 1


@dataclass(frozen=True)
class Node:
    address: Address
    type: NodeType


@dataclass(frozen=True)
class Edge:
    from_node: Node
    to_node: Node
    transaction: Transaction



class TransactionsGraph:
    nodes: set[Node] = {}
    edges: set[Edge] = {}

    def add_node(self, node: Node):
        if node not in self.nodes:
            self.nodes.add(node)
            log.info(f"Node added: {node.address} of type {node.type.name}")
        else:
            log.debug(f"Node {node.address} already exists. Skipping addition")

    # TODO REWORK THIS - is no more valid
    def add_edge(self, transaction: Transaction):
        if transaction not in self.edges:
            self.edges.add(transaction)
            log.info(f"Transaction added: {transaction.from_node.address} -> {transaction.to_node.address} with value {transaction.value}")

    def is_transaction_in_graph(self, transaction: Transaction) -> bool:
        return transaction in self.edges

    def is_transaction_in_graph_by_id(self, tx_hash) -> bool:
        for transaction in self.edges:
            if transaction.tx_hash == tx_hash:
                return True
        return False

    def get_node_by_address(self, address: str) -> Node | None:
        for node in self.nodes:
            if node.address == address:
                return node
        return None