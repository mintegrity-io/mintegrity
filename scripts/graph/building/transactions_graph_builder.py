from enum import IntEnum

from scripts.graph.building.eth.eth_transactions_scraper import get_address_interactions as eth_address_interactions
from scripts.graph.building.btc.btc_transactions_scraper import get_address_interactions as btc_address_interactions

from scripts.graph.model.transactions_graph import *

log = get_logger()

# Add a limit to the number of nodes and transactions to avoid excessive memory and API usage
MAX_NODES_PER_GRAPH_DEFAULT = 50000
MAX_TRANSACTIONS_PER_GRAPH_DEFAULT = 1000000
MAX_TRANSACTIONS_NORMAL_NODE_DEFAULT = 15000
MAX_TRANSACTIONS_ROOT_NODE_DEFAULT = 50000  # Root nodes can have more transactions due to their central role in the graph


# Track state of address processing
class AddressProcessingState(IntEnum):
    PROCESSED = 0
    PARTIALLY_PROCESSED = 1


class TargetNetwork(IntEnum):
    ETH = 0,
    BTC = 1


class TransactionsGraphBuilder:
    def __init__(self, root_addresses: set[Address], from_time: int, to_time: int, target_network: TargetNetwork,
                 max_nodes_per_graph: int = MAX_NODES_PER_GRAPH_DEFAULT,
                 max_transactions_per_graph: int = MAX_TRANSACTIONS_PER_GRAPH_DEFAULT,
                 max_transactions_normal_node: int = MAX_TRANSACTIONS_NORMAL_NODE_DEFAULT,
                 max_transactions_root_node: int = MAX_TRANSACTIONS_ROOT_NODE_DEFAULT):
        self.graph = TransactionsGraph()
        self.processed_addresses: dict[str, AddressProcessingState] = {}  # Using address string as key
        self.root_addresses: set[Address] = root_addresses
        self.from_time: int = from_time
        self.to_time: int = to_time
        self.addresses_to_process: set[Address] = set()
        self.target_network: TargetNetwork = target_network

        # Maximum limits for graph size
        self.max_nodes_per_graph = max_nodes_per_graph
        self.max_transactions_per_graph = max_transactions_per_graph
        self.max_transactions_normal_node = max_transactions_normal_node
        self.max_transactions_root_node = max_transactions_root_node

    def build_graph(self) -> TransactionsGraph:
        """
        Build the transactions graph for the given root addresses and time range.
        Handles keyboard interrupts (Ctrl+C) by returning the current state of the graph.

        :return: The built TransactionsGraph object.
        """

        try:
            # Add initial nodes
            log.info(f"Building transactions graph for {len(self.root_addresses)} root addresses")
            self.addresses_to_process: set[Address] = set([address for address in self.root_addresses])
            for address in self.root_addresses:
                self.graph.add_node_if_not_exists(address, is_root=True)
                self.add_all_interactions_for_address_and_mark_it_as_processed(address)

            iteration = 0
            while True:
                iteration += 1
                log.info(f"Graph build iteration: {iteration}. Number of nodes: {self.graph.get_number_of_nodes()}, transactions: {self.graph.get_number_of_transactions()}")
                # Get list of addresses to process. Criteria: address is in graph, but is not marked as processed yet
                self.addresses_to_process: set[Address] = self.get_new_addresses_to_process()

                addresses_to_process_copy = self.addresses_to_process.copy()
                for address in addresses_to_process_copy:
                    self.add_all_interactions_for_address_and_mark_it_as_processed(address)

                if self._termination_condition():
                    log.warning("Termination condition met. Stopping graph building.")
                    break

        except KeyboardInterrupt:
            log.warning("Keyboard interrupt detected. Returning current graph state.")
            # Mark all unprocessed addresses as partially processed
            self.mark_all_not_processed_addresses_as_partially_processed("Keyboard interrupt - graph building was stopped by user")

        log.info(f"Graph building finished. Final graph has {self.graph.get_number_of_nodes()} nodes and {self.graph.get_number_of_transactions()} transactions.")
        return self.graph

    def _termination_condition(self):
        """
        Check if the graph building process should be terminated based on the number of nodes and transactions.
        """
        if self.graph.get_number_of_nodes() > self.max_nodes_per_graph:
            reason = f"Graph has reached maximum number of nodes: {self.max_nodes_per_graph}"
            log.warning(reason)
            self.mark_all_not_processed_addresses_as_partially_processed(reason)
            return True
        if self.graph.get_number_of_transactions() > self.max_transactions_per_graph:
            reason = f"Graph has reached maximum number of transactions: {self.graph.get_number_of_transactions()}"
            log.warning(reason)
            self.mark_all_not_processed_addresses_as_partially_processed(reason)
            return True
        if len(self.addresses_to_process) == 0 and len(self.get_new_addresses_to_process()) == 0:
            log.info("No more addresses to process. Stopping graph building.")
            return True
        log.info("Current graph contains {} nodes and {} transactions".format(self.graph.get_number_of_nodes(), self.graph.get_number_of_transactions()))
        return False

    def _mark_address_as_fully_processed(self, address: Address):
        self.processed_addresses[address.address] = AddressProcessingState.PROCESSED
        self.addresses_to_process.remove(address)
        log.info(f"Processed address: {address.address}. Total processed: {len(self.processed_addresses)}")

    def mark_all_not_processed_addresses_as_partially_processed(self, reason: str):
        # Create a copy of the set to avoid "Set changed size during iteration" error
        addresses_to_process_copy = self.addresses_to_process.copy()
        for address in addresses_to_process_copy:
            self.mark_address_as_partially_processed(address, reason)

    def mark_address_as_partially_processed(self, address: Address, reason: str):
        self.processed_addresses[address.address] = AddressProcessingState.PARTIALLY_PROCESSED
        self.addresses_to_process.remove(address)
        log.info(f"Partially processed address: {address.address}. Total processed: {len(self.processed_addresses)}. Reason: {reason}")

    def is_address_processed(self, address: Address) -> bool:
        return address.address in self.processed_addresses

    def get_new_addresses_to_process(self) -> set[Address]:
        all_addresses = [node.address for node in self.graph.nodes.values()]
        addresses_to_process = set([address for address in all_addresses if not self.is_address_processed(address)])
        log.info("Found {} unique addresses in graph, {} to process".format(len(all_addresses), len(addresses_to_process)))
        return addresses_to_process

    def add_all_interactions_for_address_and_mark_it_as_processed(self, address: Address):
        # Check if the node is a root node
        node = self.graph.get_node_by_address(address)
        transaction_limit = self.max_transactions_root_node if node and node.type == NodeType.ROOT else self.max_transactions_normal_node

        try:
            interactions_with_address: set[tuple[InteractionDirection, Transaction]] = self.get_address_interactions(address, self.from_time, to_time=self.to_time, limit=transaction_limit)
            number_of_interactions_to_add: int = len(interactions_with_address)
            if number_of_interactions_to_add <= transaction_limit:
                for interaction_entry in interactions_with_address:
                    (_, transaction) = interaction_entry
                    self.graph.add_transaction(transaction)
                # Mark processed addresses to avoid infinite loops
                self._mark_address_as_fully_processed(address)
            else:
                interactions_with_address_capped: set[tuple[InteractionDirection, Transaction]] = set(list(interactions_with_address)[:transaction_limit])
                for interaction_entry in interactions_with_address_capped:
                    (_, transaction) = interaction_entry
                    self.graph.add_transaction(transaction)
                # Mark processed addresses to avoid infinite loops
                self.mark_address_as_partially_processed(address, f"Too many transactions to process. Capped at {transaction_limit}.")
        except Exception as e:
            # Log the error but continue with the next address
            log.error(f"Error fetching interactions for address {address.address}: {str(e)}")
            # Mark the address as partially processed so we don't try it again
            self.mark_address_as_partially_processed(address, f"Error fetching interactions: {str(e)}")

    def get_address_interactions(self, address, from_time, to_time, limit) -> set[tuple[InteractionDirection, Transaction]]:
        if self.target_network == TargetNetwork.ETH:
            return eth_address_interactions(address, from_time, to_time, limit=limit)
        elif self.target_network == TargetNetwork.BTC:
            return btc_address_interactions(address, from_time, to_time, limit=limit)
        else:
            raise ValueError(f"Unsupported target network: {self.target_network.name}")
