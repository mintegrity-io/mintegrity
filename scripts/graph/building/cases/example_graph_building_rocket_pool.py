from dateutil import parser

from scripts.commons import metadata
from scripts.commons.model import *
from scripts.graph.building.transactions_graph_builder import TransactionsGraphBuilder
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.util.transactions_graph_json import save_graph_to_json

FROM_TIME = int(parser.parse("2025-03-25T00:00:00Z").timestamp())
TO_TIME = int(parser.parse("2025-05-24T00:00:00Z").timestamp())
GRAPH_PATH = "../../files/rocket_pool_graph_60_days.json"

metadata.init()

contracts: set[SmartContract] = {SmartContract(address=Address("0xDD3f50F8A6CafbE9b31a427582963f465E745AF8", AddressType.CONTRACT))}

graph: TransactionsGraph = TransactionsGraphBuilder(contracts, FROM_TIME, TO_TIME).build_graph()
print(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

save_graph_to_json(graph, GRAPH_PATH)
