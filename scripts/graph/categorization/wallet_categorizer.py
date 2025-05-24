from dataclasses import field, dataclass
from datetime import datetime, timezone
from collections import defaultdict
from enum import StrEnum
from typing import Dict, List, Optional

from scripts.commons.logging_config import get_logger
from scripts.graph.model.transactions_graph import TransactionsGraph, Node, NodeType

log = get_logger()


class WalletType(StrEnum):
    # Major institutional entities
    EXCHANGE = "exchange"  # High volume trading platforms
    MARKET_MAKER = "market_maker"  # Balance liquidity providers
    CUSTODIAN = "custodian"  # Long-term holders with minimal outflow

    # Special wallets
    WHALE = "whale"  # Large holders with significant market impact
    DEVELOPER = "developer"  # Wallets associated with project development
    DAO_TREASURY = "dao_treasury"  # DAO controlled treasuries

    # Behavioral patterns
    TRADER = "trader"  # Active trading wallets
    BRIDGE_OR_MIXER = "bridge_or_mixer"  # Cross-chain bridges or privacy tools
    YIELD_FARMER = "yield_farmer"  # Active in DeFi with multiple small transactions

    # Others
    DUST_WALLET = "dust_wallet"  # Minimal activity and value
    NEW_WALLET = "new_wallet"  # Recently created with few transactions
    INACTIVE_WALLET = "inactive_wallet"  # Long periods without transactions
    REGULAR_USER = "regular_user"  # Standard user wallet


@dataclass
class WalletStats:
    # Basic metrics
    total_usd_in: float = 0
    total_usd_out: float = 0
    tx_count_in: int = 0
    tx_count_out: int = 0

    # Temporal metrics
    first_tx_timestamp: Optional[datetime] = None
    last_tx_timestamp: Optional[datetime] = None

    # Network metrics
    unique_counterparties: set = field(default_factory=set)
    repeated_interactions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    contracts_interacted_with: set = field(default_factory=set)

    # Transaction patterns
    tx_size_distribution: List[float] = field(default_factory=list)
    periodic_tx_patterns: Dict[str, int] = field(default_factory=dict)  # e.g., 'daily', 'weekly'

    # Advanced metrics
    max_tx_value_usd: float = 0
    avg_tx_value_usd: float = 0
    in_out_ratio: float = 0  # Calculated after all transactions are processed

    def finalize(self):
        """Calculate derived metrics after all transactions are processed"""
        total_tx = self.tx_count_in + self.tx_count_out
        if total_tx > 0:
            self.avg_tx_value_usd = (self.total_usd_in + self.total_usd_out) / total_tx

        if self.total_usd_out > 0:
            self.in_out_ratio = self.total_usd_in / self.total_usd_out
        elif self.total_usd_in > 0:  # Only incoming, no outgoing
            self.in_out_ratio = float('inf')
        else:
            self.in_out_ratio = 0

    def get_activity_duration_days(self) -> float:
        """Get the duration between first and last transaction in days"""
        if not self.first_tx_timestamp or not self.last_tx_timestamp:
            return 0

        duration = self.last_tx_timestamp - self.first_tx_timestamp
        return duration.total_seconds() / (60 * 60 * 24)  # Convert to days


def categorize_wallet(graph: TransactionsGraph, node: Node) -> WalletType:
    """
    Categorize a wallet node based on its transaction patterns

    Args:
        graph: The transaction graph containing the node
        node: The wallet node to categorize

    Returns:
        A WalletType indicating the wallet's category
    """
    if node.type != NodeType.WALLET:
        raise ValueError(f"Node {node.address.address} is not a wallet node")

    stats = WalletStats()
    contract_interactions = 0

    # Collect transaction statistics
    for (from_addr, to_addr), edge in graph.edges.items():
        if node.address.address == from_addr:
            stats.tx_count_out += len(edge.transactions)

            # Check if counterparty is a contract
            to_node = graph.nodes.get(to_addr)
            if to_node and to_node.type == NodeType.CONTRACT:
                contract_interactions += len(edge.transactions)
                stats.contracts_interacted_with.add(to_addr)

            # Process outgoing transactions
            for tx in edge.transactions.values():
                tx_value_usd = tx.value_usd
                stats.total_usd_out += tx_value_usd
                stats.tx_size_distribution.append(tx_value_usd)
                stats.max_tx_value_usd = max(stats.max_tx_value_usd, tx_value_usd)

                # Parse timestamp for temporal analysis
                try:
                    tx_time = datetime.fromisoformat(tx.timestamp.replace('Z', '+00:00'))
                    if not stats.first_tx_timestamp or tx_time < stats.first_tx_timestamp:
                        stats.first_tx_timestamp = tx_time
                    if not stats.last_tx_timestamp or tx_time > stats.last_tx_timestamp:
                        stats.last_tx_timestamp = tx_time
                except (ValueError, AttributeError):
                    log.warning(f"Could not parse timestamp {tx.timestamp}")

                # Track repeated interactions
                stats.unique_counterparties.add(to_addr)
                stats.repeated_interactions[to_addr] += 1

        elif node.address.address == to_addr:
            stats.tx_count_in += len(edge.transactions)

            # Check if counterparty is a contract
            from_node = graph.nodes.get(from_addr)
            if from_node and from_node.type == NodeType.CONTRACT:
                stats.contracts_interacted_with.add(from_addr)

            # Process incoming transactions
            for tx in edge.transactions.values():
                tx_value_usd = tx.value_usd
                stats.total_usd_in += tx_value_usd
                stats.tx_size_distribution.append(tx_value_usd)
                stats.max_tx_value_usd = max(stats.max_tx_value_usd, tx_value_usd)

                # Parse timestamp for temporal analysis
                try:
                    tx_time = datetime.fromisoformat(tx.timestamp.replace('Z', '+00:00'))
                    if not stats.first_tx_timestamp or tx_time < stats.first_tx_timestamp:
                        stats.first_tx_timestamp = tx_time
                    if not stats.last_tx_timestamp or tx_time > stats.last_tx_timestamp:
                        stats.last_tx_timestamp = tx_time
                except (ValueError, AttributeError):
                    log.warning(f"Could not parse timestamp {tx.timestamp}")

                # Track repeated interactions
                stats.unique_counterparties.add(from_addr)
                stats.repeated_interactions[from_addr] += 1

    # Calculate derived metrics
    stats.finalize()

    # Get key metrics
    total_tx = stats.tx_count_in + stats.tx_count_out
    total_value_usd = stats.total_usd_in + stats.total_usd_out
    counterparties = len(stats.unique_counterparties)
    activity_days = stats.get_activity_duration_days()
    unique_contracts = len(stats.contracts_interacted_with)

    # High frequency repeating interactions
    high_frequency_interactions = sum(1 for count in stats.repeated_interactions.values() if count > 10)

    # Average interactions per counterparty
    avg_interactions_per_counterparty = total_tx / max(1, counterparties)

    # Contract interaction ratio
    contract_interaction_ratio = contract_interactions / max(1, total_tx)

    # Debug logging
    log.debug(f"Address: {node.address.address}")
    log.debug(f"Total TX: {total_tx}, Value USD: {total_value_usd:.2f}, Counterparties: {counterparties}")
    log.debug(f"Activity Days: {activity_days:.2f}, Contracts: {unique_contracts}, In/Out Ratio: {stats.in_out_ratio:.2f}")

    # Advanced heuristics for wallet classification

    # EXCHANGE detection
    if (total_tx > 1000 and counterparties > 500 and activity_days > 30 and stats.in_out_ratio >= 0.85 and stats.in_out_ratio <= 1.15):
        return WalletType.EXCHANGE

    # MARKET_MAKER detection - balanced flows, high frequency, many counterparties
    if (total_tx > 500 and
            counterparties > 100 and
            0.9 <= stats.in_out_ratio <= 1.1 and
            activity_days > 14 and
            high_frequency_interactions > 50):
        return WalletType.MARKET_MAKER

    # CUSTODIAN detection - large holdings, mostly inflows, few outflows
    if (total_value_usd > 5_000_000 and
            stats.in_out_ratio > 5.0 and
            stats.tx_count_out < 0.2 * stats.tx_count_in and
            activity_days > 60):
        return WalletType.CUSTODIAN

    # WHALE detection - large holdings
    if (total_value_usd > 1_000_000 and
            stats.max_tx_value_usd > 100_000):
        return WalletType.WHALE

    # DEVELOPER detection - many contract interactions, regular activity
    if (unique_contracts > 10 and
            contract_interaction_ratio > 0.7 and
            activity_days > 30 and
            total_tx > 50):
        return WalletType.DEVELOPER

    # DAO_TREASURY detection - large holdings, periodic transactions, interaction with governance contracts
    if (total_value_usd > 500_000 and
            activity_days > 30 and
            contract_interaction_ratio > 0.6 and
            len(set(tx for tx in stats.tx_size_distribution if abs(stats.avg_tx_value_usd - tx) < 0.1 * stats.avg_tx_value_usd)) > 10):
        return WalletType.DAO_TREASURY

    # TRADER detection - frequent transactions, varied counterparties, balanced flows
    if (total_tx > 50 and
            activity_days > 7 and
            counterparties > 10 and
            avg_interactions_per_counterparty < 5 and
            0.3 <= stats.in_out_ratio <= 3.0):
        return WalletType.TRADER

    # BRIDGE_OR_MIXER detection - balanced flows, consistent transaction sizes
    if (total_tx > 50 and
            abs(stats.total_usd_in - stats.total_usd_out) < 0.15 * total_value_usd and
            activity_days > 7 and
            len(set(round(tx / 100) * 100 for tx in stats.tx_size_distribution)) < 0.3 * len(stats.tx_size_distribution)):
        return WalletType.BRIDGE_OR_MIXER

    # YIELD_FARMER detection - many small transactions with DeFi contracts
    if (unique_contracts > 5 and
            contract_interaction_ratio > 0.8 and
            total_tx > 30 and
            stats.avg_tx_value_usd < 5000 and
            activity_days > 7):
        return WalletType.YIELD_FARMER

    # DUST_WALLET detection - minimal activity and value
    if (total_tx < 5 and
            total_value_usd < 100):
        return WalletType.DUST_WALLET

    # NEW_WALLET detection - recently created with few transactions
    if (activity_days < 7 and
            total_tx < 10):
        return WalletType.NEW_WALLET

    # INACTIVE_WALLET detection - no recent transactions
    if stats.last_tx_timestamp:
        # Make sure current_time is timezone-aware to match stats.last_tx_timestamp
        current_time = datetime.now(timezone.utc)
        days_since_last_tx = (current_time - stats.last_tx_timestamp).total_seconds() / (60 * 60 * 24)
        if days_since_last_tx > 60:  # No activity for 60+ days
            return WalletType.INACTIVE_WALLET

    # Default case
    return WalletType.REGULAR_USER
