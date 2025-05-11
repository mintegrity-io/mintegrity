from scripts.commons.logging_config import get_logger
log = get_logger()

class TransactionsGraphBuilder:
    def __init__(self):
        self.graph = TransactionsGraph()

    def add_transaction(self, transaction: Transaction):
        # Add nodes to the graph
        from_node = Node(transaction.address_from, NodeType.WALLET)
        to_node = Node(transaction.address_to, NodeType.CONTRACT)

        self.graph.add_node(from_node)
        self.graph.add_node(to_node)

        # Add the transaction as an edge in the graph
        self.graph.add_edge(transaction)

    def get_graph(self):
        return self.graph