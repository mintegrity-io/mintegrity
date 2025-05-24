from dataclasses import field, dataclass

from scripts.graph.model.transactions_graph import TransactionsGraph, Node, NodeType

from dataclasses import field, dataclass
from enum import StrEnum

from scripts.graph.model.transactions_graph import TransactionsGraph, Node


class WalletType(StrEnum):
    EXCHANGE = "exchange"
    WHALE = "whale"
    DUST_WALLET = "dust_wallet"
    BRIDGE_OR_MIXER = "bridge_or_mixer"
    USER = "user"


@dataclass
class WalletStats:
    total_usd_in: float = 0
    total_usd_out: float = 0
    tx_count_in: int = 0
    tx_count_out: int = 0
    unique_counterparties: set = field(default_factory=set)



def categorize_wallet(graph: TransactionsGraph, node: Node) -> WalletType:
    if node.type != NodeType.WALLET:
        raise ValueError(f"Node {node.address.address} is not a wallet node")

    stats = WalletStats()

    for (from_addr, to_addr), edge in graph.edges.items():
        if node.address.address == from_addr:
            stats.tx_count_out += len(edge.transactions)
            for tx in edge.transactions.values():
                stats.total_usd_out += tx.value_usd
                stats.unique_counterparties.add(to_addr)
        elif node.address.address == to_addr:
            stats.tx_count_in += len(edge.transactions)
            for tx in edge.transactions.values():
                stats.total_usd_in += tx.value_usd
                stats.unique_counterparties.add(from_addr)

    total_tx = stats.tx_count_in + stats.tx_count_out
    total_value_usd = stats.total_usd_in + stats.total_usd_out
    counterparties = len(stats.unique_counterparties)

    # Heuristics
    if total_tx > 1000 and counterparties > 500:
        return WalletType.EXCHANGE
    elif total_value_usd > 1_000_000 and total_tx < 100:
        return WalletType.WHALE
    elif total_tx < 3 and total_value_usd < 100:
        return WalletType.DUST_WALLET
    elif total_tx > 500 and abs(stats.total_usd_in - stats.total_usd_out) < 0.1 * total_value_usd:
        return WalletType.BRIDGE_OR_MIXER
    else:
        return WalletType.USER

