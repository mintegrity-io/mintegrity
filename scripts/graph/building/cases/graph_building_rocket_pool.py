from dateutil import parser

from scripts.commons import metadata
from scripts.commons.model import *
from scripts.graph.building.transactions_graph_builder import TransactionsGraphBuilder
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.util.transactions_graph_json import save_graph_to_json

FROM_TIME = int(parser.parse("2025-02-25T00:00:00Z").timestamp())
TO_TIME = int(parser.parse("2025-05-24T00:00:00Z").timestamp())
GRAPH_PATH = "../../../../cases/rocketpool/files/rocket_pool_full_graph_90_days.json"

metadata.init()

contracts: set[SmartContract] = {SmartContract(address=Address("0xdd3f50f8a6cafbe9b31a427582963f465e745af8", AddressType.CONTRACT)),
                                 SmartContract(address=Address("0x1d8f8f00cfa6758d7bE78336684788Fb0ee0Fa46", AddressType.CONTRACT))}

graph: TransactionsGraph = TransactionsGraphBuilder(contracts, FROM_TIME, TO_TIME).build_graph()
print(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

save_graph_to_json(graph, GRAPH_PATH)
