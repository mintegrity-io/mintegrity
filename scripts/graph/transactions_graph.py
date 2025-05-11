from dataclasses import dataclass
from enum import IntEnum

from scripts.commons.logging_config import get_logger

log = get_logger()


class NodeType(IntEnum):
    WALLET = 0
    CONTRACT = 1


@dataclass(frozen=True)
class Node:
    address: str
    type: NodeType


@dataclass(frozen=True)
class Transaction:
    tx_hash: str
    from_node: Node
    to_node: Node
    value: float
    timestamp: int


class TransactionsGraph:
    nodes: set[Node] = {}
    edges: set[Transaction] = {}

    def add_node(self, node: Node):
        if node not in self.nodes:
            self.nodes.add(node)
            log.info(f"Node added: {node.address} of type {node.type.name}")
        else:
            log.debug(f"Node {node.address} already exists. Skipping addition")

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