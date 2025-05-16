from dataclasses import dataclass
from enum import IntEnum

from scripts.commons.logging_config import get_logger
from scripts.commons.model import *

log = get_logger()


class NodeType(IntEnum):
    ROOT = 1
    WALLET = 2
    CONTRACT = 3


@dataclass(frozen=True)
class Node:
    address: Address
    type: NodeType

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address.to_dict(),
            "type": self.type.value
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Node':
        return cls(
            address=Address.from_dict(data["address"]),
            type=NodeType(data["type"])
        )


@dataclass
class Edge:
    from_node: Node
    to_node: Node
    transactions: set[Transaction]  # it is possible and likely, that multiple transactions exist between the same nodes, e.g. many interactions between two wallets

    def __hash__(self):
        # Only hash based on from_node and to_node which are constant
        return hash((self.from_node, self.to_node))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        # Two edges are equal if they connect the same nodes
        return self.from_node == other.from_node and self.to_node == other.to_node

    def to_dict(self) -> dict[str, Any]:
            return {
                "from_node": self.from_node.to_dict(),
                "to_node": self.to_node.to_dict(),
                "transactions": [tx.to_dict() for tx in self.transactions]
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any], node_map: dict[str, Node] = None) -> 'Edge':
        if node_map:
            # Use existing nodes if provided
            from_address = Address.from_dict(data["from_node"]["address"])
            to_address = Address.from_dict(data["to_node"]["address"])
            from_node = node_map.get(from_address.address)
            to_node = node_map.get(to_address.address)
        else:
            from_node = Node.from_dict(data["from_node"])
            to_node = Node.from_dict(data["to_node"])

        transactions = {Transaction.from_dict(tx) for tx in data["transactions"]}
        return cls(from_node=from_node, to_node=to_node, transactions=transactions)

    def add_transaction(self, transaction: Transaction):
        if transaction not in self.transactions:
            self.transactions.add(transaction)
            log.info(f"Transaction added to edge: {transaction.transaction_hash} from {self.from_node.address} to {self.to_node.address}")
        else:
            log.debug(f"Transaction {transaction.transaction_hash} already exists in edge. Skipping addition")

    def get_total_transactions_value(self) -> float:
        return sum(transaction.value for transaction in self.transactions)


class TransactionsGraph:
    nodes: set[Node] = set()
    edges: set[Edge] = set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TransactionsGraph':
        graph = cls()

        # First create all nodes
        node_map = {}
        for node_data in data["nodes"]:
            node = Node.from_dict(node_data)
            graph.nodes.add(node)
            node_map[node.address.address] = node

        # Then create edges using the node references
        for edge_data in data["edges"]:
            edge = Edge.from_dict(edge_data, node_map)
            graph.edges.add(edge)

        return graph

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
        """
        Add a transaction to the graph. If the transaction already exists, skip it.
        Create all corresponding nodes and edges if they do not exist.
        """
        if self.is_transaction_in_graph(transaction):
            log.debug(f"Transaction {transaction.transaction_hash} already exists in graph. Skipping addition")
            return

        from_node: Node = self.add_node_if_not_exists(transaction.address_from)
        to_node: Node = self.add_node_if_not_exists(transaction.address_to)

        edge: Edge = self.add_edge_if_not_exists(from_node, to_node)
        edge.add_transaction(transaction)

    def get_number_of_nodes(self) -> int:
        return len(self.nodes)

    def get_number_of_edges(self) -> int:
        return len(self.edges)

    def get_number_of_transactions(self) -> int:
        return sum(len(edge.transactions) for edge in self.edges)
