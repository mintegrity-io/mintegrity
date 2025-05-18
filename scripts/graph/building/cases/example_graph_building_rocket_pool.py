from dateutil import parser

from scripts.commons import initialize_metadata
from scripts.commons.model import *
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.building.transactions_graph_builder import TransactionsGraphBuilder
from scripts.graph.util.transactions_graph_json import save_graph_to_json

FROM_TIME = int(parser.parse("2025-05-01T00:00:00Z").timestamp())
TO_TIME = int(parser.parse("2025-05-11T00:00:00Z").timestamp())
GRAPH_PATH = "../../files/rocket_pool_graph_10_days.json"

initialize_metadata.init()

contracts: set[SmartContract] = {SmartContract(address=Address("0xDD3f50F8A6CafbE9b31a427582963f465E745AF8", AddressType.CONTRACT))}

graph: TransactionsGraph = TransactionsGraphBuilder(contracts, FROM_TIME, TO_TIME).build_graph()
print(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

save_graph_to_json(graph, GRAPH_PATH)
