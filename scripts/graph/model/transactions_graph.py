from scripts.commons.model import *

log = get_logger()


class NodeType(StrEnum):
    ROOT = "root"
    WALLET = "wallet"
    CONTRACT = "contract"


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
    transactions: dict[str, Transaction]  # Using transaction_hash as key for efficient lookups

    def __init__(self, from_node: Node, to_node: Node, transactions=None):
        self.from_node = from_node
        self.to_node = to_node
        self.transactions = transactions if transactions is not None else {}

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
            "transactions": [tx.to_dict() for tx in self.transactions.values()]
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

        # Convert list of transactions to dictionary with hash as key
        transactions_dict = {}
        for tx_data in data["transactions"]:
            tx = Transaction.from_dict(tx_data)
            transactions_dict[tx.transaction_hash] = tx

        return cls(from_node=from_node, to_node=to_node, transactions=transactions_dict)

    def add_transaction(self, transaction: Transaction):
        if transaction.transaction_hash not in self.transactions:
            self.transactions[transaction.transaction_hash] = transaction
            log.info(f"Transaction added to edge: {transaction.transaction_hash} from {self.from_node.address.address} to {self.to_node.address.address}")
        else:
            log.debug(f"Transaction {transaction.transaction_hash} already exists in edge. Skipping addition")

    def get_total_transactions_value_usd(self) -> float:
        return sum(transaction.value_usd for transaction in self.transactions.values())


class TransactionsGraph:
    def __init__(self):
        # Use dictionaries for faster lookups
        self.nodes: dict[str, Node] = {}  # key: address string
        self.edges: dict[tuple[str, str], Edge] = {}  # key: (from_address, to_address)
        self.tx_to_edge: dict[str, tuple[str, str]] = {}  # key: transaction_hash, value: edge key

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TransactionsGraph':
        graph = cls()

        # First create all nodes
        for node_data in data["nodes"]:
            node = Node.from_dict(node_data)
            graph.nodes[node.address.address] = node

        # Then create edges using the node references
        for edge_data in data["edges"]:
            from_address = Address.from_dict(edge_data["from_node"]["address"])
            to_address = Address.from_dict(edge_data["to_node"]["address"])

            from_node = graph.nodes[from_address.address]
            to_node = graph.nodes[to_address.address]

            edge = Edge.from_dict(edge_data, graph.nodes)
            edge_key = (from_node.address.address, to_node.address.address)
            graph.edges[edge_key] = edge

            # Map transaction hashes to edge key for quick lookups
            for tx_hash in edge.transactions:
                graph.tx_to_edge[tx_hash] = edge_key

        return graph

    def add_node_if_not_exists(self, address: Address, is_root: bool = False) -> Node:
        if address.address not in self.nodes:
            if is_root:
                node_type = NodeType.ROOT
            else:
                if address.type == AddressType.WALLET:
                    node_type = NodeType.WALLET
                elif address.type == AddressType.CONTRACT:
                    node_type = NodeType.CONTRACT
                else:
                    raise ValueError(f"Unknown address type: {address.type}")
            new_node = Node(address, node_type)
            self.nodes[address.address] = new_node
            log.info(f"Node added: {new_node.address.address} of type {new_node.type.name}")
        else:
            log.debug(f"Node with address {address.address} in graph already exists. Skipping addition")

        return self.nodes[address.address]

    def add_edge_if_not_exists(self, from_node: Node, to_node: Node) -> Edge:
        edge_key = (from_node.address.address, to_node.address.address)
        if edge_key not in self.edges:
            new_edge = Edge(from_node, to_node)
            self.edges[edge_key] = new_edge
            log.info(f"Edge added: {from_node.address.address} -> {to_node.address.address}")
        else:
            log.debug(f"Edge between {from_node.address.address} and {to_node.address.address} already exists. Skipping addition")

        return self.edges[edge_key]

    def is_transaction_in_graph(self, transaction: Transaction) -> bool:
        return transaction.transaction_hash in self.tx_to_edge

    def is_node_in_graph_by_address(self, address: Address) -> bool:
        return address.address in self.nodes

    def get_node_by_address(self, address: Address) -> Node:
        return self.nodes.get(address.address)

    def is_edge_between_nodes_in_graph(self, from_node: Node, to_node: Node) -> bool:
        edge_key = (from_node.address.address, to_node.address.address)
        return edge_key in self.edges

    def get_edge_by_nodes(self, from_node: Node, to_node: Node) -> Edge:
        edge_key = (from_node.address.address, to_node.address.address)
        return self.edges.get(edge_key)

    def add_transaction(self, transaction: Transaction):
        """
        Add a transaction to the graph. If the transaction already exists, skip it.
        Create all corresponding nodes and edges if they do not exist.
        """
        if transaction.transaction_hash in self.tx_to_edge:
            log.debug(f"Transaction {transaction.transaction_hash} already exists in graph. Skipping addition")
            return

        from_node = self.add_node_if_not_exists(transaction.address_from)
        to_node = self.add_node_if_not_exists(transaction.address_to)

        edge = self.add_edge_if_not_exists(from_node, to_node)
        edge.add_transaction(transaction)

        # Add to transaction lookup map
        edge_key = (from_node.address.address, to_node.address.address)
        self.tx_to_edge[transaction.transaction_hash] = edge_key

    def get_number_of_nodes(self) -> int:
        return len(self.nodes)

    def get_number_of_edges(self) -> int:
        return len(self.edges)

    def get_number_of_transactions(self) -> int:
        return len(self.tx_to_edge)
