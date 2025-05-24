from dataclasses import field, dataclass
from enum import StrEnum

from scripts.graph.model.transactions_graph import TransactionsGraph, Node


class ContractType(StrEnum):
    DEX = "dex"
    LENDING = "lending"
    NFT_MARKETPLACE = "nft_marketplace"
    STAKING = "staking"
    TOKEN = "token"
    BRIDGE = "bridge"
    GENERIC = "generic"


@dataclass
class ContractStats:
    total_usd_in: float = 0
    total_usd_out: float = 0
    tx_count_in: int = 0
    tx_count_out: int = 0
    unique_senders: set = field(default_factory=set)
    unique_receivers: set = field(default_factory=set)


def categorize_contract(graph: TransactionsGraph, node: Node) -> ContractType:
    """
    Categorize a contract node based on its transaction patterns.

    Args:
        graph: The transaction graph containing the node
        node: The contract node to categorize

    Returns:
        A ContractType indicating the contract's category
    """
    stats = ContractStats()

    # Collect statistics about the contract's transaction patterns
    for (from_addr, to_addr), edge in graph.edges.items():
        if node.address.address == from_addr:
            stats.tx_count_out += len(edge.transactions)
            for tx in edge.transactions.values():
                stats.total_usd_out += tx.value_usd
                stats.unique_receivers.add(to_addr)
        elif node.address.address == to_addr:
            stats.tx_count_in += len(edge.transactions)
            for tx in edge.transactions.values():
                stats.total_usd_in += tx.value_usd
                stats.unique_senders.add(from_addr)

    total_tx = stats.tx_count_in + stats.tx_count_out
    total_value_usd = stats.total_usd_in + stats.total_usd_out
    unique_senders_count = len(stats.unique_senders)
    unique_receivers_count = len(stats.unique_receivers)

    # Heuristics for contract classification

    # DEX patterns: many unique senders and receivers, similar in/out value
    if unique_senders_count > 100 and unique_receivers_count > 100:
        if 0.8 < (stats.total_usd_in / stats.total_usd_out) < 1.2:
            return ContractType.DEX

    # Lending platforms: ratio of in/out transactions fluctuates over time
    if unique_senders_count > 50 and total_tx > 500:
        if stats.total_usd_in > 0 and stats.total_usd_out > 0:
            return ContractType.LENDING

    # Staking contracts: typically receive more than they send out
    if stats.tx_count_in > stats.tx_count_out * 1.5 and total_value_usd > 100000:
        return ContractType.STAKING

    # Token contracts: very high transaction count
    if total_tx > 1000 and (unique_senders_count > 200 or unique_receivers_count > 200):
        return ContractType.TOKEN

    # Bridge: balanced flow with high value, fewer transactions
    if total_tx < 1000 and total_value_usd > 500000 and 0.7 < (stats.total_usd_in / (stats.total_usd_out + 0.001)) < 1.3:
        return ContractType.BRIDGE

    # NFT Marketplace: numerous small transactions
    if total_tx > 200 and unique_receivers_count > 50 and stats.total_usd_out / (stats.tx_count_out + 0.001) < 1000:
        return ContractType.NFT_MARKETPLACE

    # Default category if no specific pattern is detected
    return ContractType.GENERIC
