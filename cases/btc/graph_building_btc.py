from dateutil import parser

from scripts.commons import metadata
from scripts.commons.model import *
from scripts.graph.building.transactions_graph_builder import TransactionsGraphBuilder, TargetNetwork
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.util.transactions_graph_json import save_graph_to_json

days = 10

TO_TIME = int(parser.parse("2025-06-01T00:00:00Z").timestamp())
FROM_TIME = TO_TIME - (days * 24 * 60 * 60)

GRAPH_NAME = f"btc_full_graph_{days}_days"
GRAPH_PATH = f"./files/{GRAPH_NAME}.json"

metadata.init()

root_wallets: set[Address] = {Address("0xdd3f50f8a6cafbe9b31a427582963f465e745af8", type=AddressType.WALLET),
                              Address("0x1d8f8f00cfa6758d7bE78336684788Fb0ee0Fa46", type=AddressType.WALLET)}

graph: TransactionsGraph = TransactionsGraphBuilder(root_wallets, FROM_TIME, TO_TIME, TargetNetwork.BTC).build_graph()
print(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

save_graph_to_json(graph, GRAPH_PATH)
