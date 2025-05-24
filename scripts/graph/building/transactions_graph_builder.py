from scripts.commons.transactions_metadata_scraper import get_address_interactions
from scripts.graph.model.transactions_graph import *

log = get_logger()

# Add a limit to the number of nodes and transactions to avoid excessive memory and API usage
MAX_NODES_PER_GRAPH = 5000
MAX_TRANSACTIONS_PER_GRAPH = 10000
MAX_TRANSACTIONS_PER_NODE = 1000


# Track state of address processing
class AddressProcessingState(IntEnum):
    PROCESSED = 0
    PARTIALLY_PROCESSED = 1


class TransactionsGraphBuilder:
    def __init__(self, root_contracts: set[SmartContract], from_time: int, to_time: int):
        self.graph = TransactionsGraph()
        self.processed_addresses: dict[str, AddressProcessingState] = {}  # Using address string as key
        self.root_contracts: set[SmartContract] = root_contracts
        self.from_time: int = from_time
        self.to_time: int = to_time
        self.addresses_to_process: set[Address] = set()

    def build_graph(self) -> TransactionsGraph:
        """
        Build the transactions graph for the given root contracts and time range.

        :return: The built TransactionsGraph object.
        """

        # Add initial nodes
        log.info(f"Building transactions graph for {len(self.root_contracts)} root contracts")
        self.addresses_to_process: set[Address] = set([contract.address for contract in self.root_contracts])
        for contract in self.root_contracts:
            self.graph.add_node_if_not_exists(contract.address, is_root=True)
            self.add_all_interactions_for_address_and_mark_it_as_processed(contract.address)

        iteration = 0
        while True:
            iteration += 1
            log.info(f"Graph build iteration: {iteration}")
            # Get list of addresses to process. Criteria: address is in graph, but is not marked as processed yet
            self.addresses_to_process: set[Address] = self.get_new_addresses_to_process()

            addresses_to_process_copy = self.addresses_to_process.copy()
            for address in addresses_to_process_copy:
                self.add_all_interactions_for_address_and_mark_it_as_processed(address)

            if self._termination_condition():
                break

        log.warning("Termination condition met. Stopping graph building.")
        return self.graph

    def _termination_condition(self):
        """
        Check if the graph building process should be terminated based on the number of nodes and transactions.
        """
        if self.graph.get_number_of_nodes() > MAX_NODES_PER_GRAPH:
            self.mark_all_not_processed_addresses_as_partially_processed(f"Graph has reached maximum number of nodes: {MAX_NODES_PER_GRAPH}")
            return True
        if self.graph.get_number_of_transactions() > MAX_TRANSACTIONS_PER_GRAPH:
            self.mark_all_not_processed_addresses_as_partially_processed(f"Graph has reached maximum number of transactions: {MAX_TRANSACTIONS_PER_GRAPH}")
            return True
        if len(self.addresses_to_process) == 0:
            log.info("No more addresses to process. Stopping graph building.")
            return True
        log.info("Current graph contains {} nodes and {} transactions".format(self.graph.get_number_of_nodes(), self.graph.get_number_of_transactions()))
        return False

    def _mark_address_as_fully_processed(self, address: Address):
        self.processed_addresses[address.address] = AddressProcessingState.PROCESSED
        self.addresses_to_process.remove(address)
        log.info(f"Processed address: {address.address}. Total processed: {len(self.processed_addresses)}")

    def mark_all_not_processed_addresses_as_partially_processed(self, reason: str):
        for address in self.addresses_to_process:
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
        interactions_with_address: set[tuple[InteractionDirection, Transaction]] = get_address_interactions(address, self.from_time, self.to_time)
        number_of_interactions_to_add: int = len(interactions_with_address)
        if number_of_interactions_to_add <= MAX_TRANSACTIONS_PER_NODE:
            for interaction_entry in interactions_with_address:
                (_, transaction) = interaction_entry
                self.graph.add_transaction(transaction)
            # Mark processed addresses to avoid infinite loops
            self._mark_address_as_fully_processed(address)
        else:
            interactions_with_address_capped: set[tuple[InteractionDirection, Transaction]] = set(list(interactions_with_address)[:MAX_TRANSACTIONS_PER_NODE])
            for interaction_entry in interactions_with_address_capped:
                (_, transaction) = interaction_entry
                self.graph.add_transaction(transaction)
            # Mark processed addresses to avoid infinite loops
            self.mark_address_as_partially_processed(address, f"Too many transactions to process. Capped at {MAX_TRANSACTIONS_PER_NODE}.")
