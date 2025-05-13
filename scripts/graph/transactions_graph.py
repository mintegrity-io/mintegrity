from dataclasses import dataclass
from enum import IntEnum

from scripts.commons.logging_config import get_logger
from scripts.commons.model import *

log = get_logger()


class NodeType(IntEnum):
    ROOT = 1
    WALLET = 2
    CONTRACT = 3


@dataclass
class Node:
    address: Address
    type: NodeType


@dataclass
class Edge:
    from_node: Node
    to_node: Node
    transactions: set[Transaction]  # it is possible and likely, that multiple transactions exist between the same nodes, e.g. many interactions between two wallets

    def add_transaction(self, transaction: Transaction):
        if transaction not in self.transactions:
            self.transactions.add(transaction)
            log.info(f"Transaction added to edge: {transaction.transaction_hash} from {self.from_node.address} to {self.to_node.address}")
        else:
            log.debug(f"Transaction {transaction.transaction_hash} already exists in edge. Skipping addition")

    def get_total_transactions_value(self) -> float:
        return sum(transaction.value for transaction in self.transactions)


class TransactionsGraph:
    nodes: set[Node] = {}
    edges: set[Edge] = {}

    def add_node_if_not_exists(self, address: Address, is_root: bool = False) -> Node:
        if not self.is_node_in_graph_by_address(address):
            if is_root:
                node_type = NodeType.ROOT
            else:
                node_type = NodeType.WALLET if address.type == AddressType.WALLET else NodeType.CONTRACT
            new_node = Node(address, node_type)
            self.nodes.add(new_node)
            log.info(f"Node added: {new_node.address} of type {new_node.type.name}")
        else:
            log.debug(f"Node with address {address} in graph already exists. Skipping addition")

        return self.get_node_by_address(address)

    def add_edge_if_not_exists(self, from_node: Node, to_node: Node) -> Edge:
        if not self.is_edge_between_nodes_in_graph(from_node, to_node):
            new_edge = Edge(from_node, to_node, set())
            self.edges.add(new_edge)
            log.info(f"Edge added: {from_node.address} -> {to_node.address}")
        else:
            log.debug(f"Edge between {from_node.address} and {to_node.address} already exists. Skipping addition")

        return self.get_edge_by_nodes(from_node, to_node)

    def is_transaction_in_graph(self, transaction: Transaction) -> bool:
        for edge in self.edges:
            if transaction in edge.transactions:
                return True
        return False

    def is_node_in_graph_by_address(self, address: Address) -> bool:
        return self.get_node_by_address(address) is not None

    def get_node_by_address(self, address: Address) -> Node | None:
        for node in self.nodes:
            if node.address == address:
                return node
        return None

    def is_edge_between_nodes_in_graph(self, from_node, to_node) -> bool:
        return self.get_edge_by_nodes(from_node, to_node) is not None

    def get_edge_by_nodes(self, from_node, to_node) -> Edge | None:
        for edge in self.edges:
            if edge.from_node == from_node and edge.to_node == to_node:
                return edge
        return None

    def add_transaction(self, transaction: Transaction):
        if self.is_transaction_in_graph(transaction):
            log.debug(f"Transaction {transaction.transaction_hash} already exists in graph. Skipping addition")
            return

        from_node: Node = self.add_node_if_not_exists(transaction.address_from)
        to_node: Node = self.add_node_if_not_exists(transaction.address_to)

        edge: Edge = self.add_edge_if_not_exists(from_node, to_node)
        edge.add_transaction(transaction)


