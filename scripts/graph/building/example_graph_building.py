from dateutil import parser

from scripts.commons import metadata
from scripts.commons.transactions_metadata_scraper import get_smart_contracts_by_issuer
from scripts.commons.model import *
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.building.transactions_graph_builder import TransactionsGraphBuilder
from scripts.graph.util.transactions_graph_json import save_graph_to_json

metadata.init()

CONTRACT_ISSUER_ADDRESS = Address("0x8cfae48fb3e54e143e5454ca2784b7bf3a0dc0d4", AddressType.WALLET)
FROM_TIME = int(parser.parse("2025-05-04T00:00:00Z").timestamp())
TO_TIME = int(parser.parse("2025-05-05T17:00:00Z").timestamp())
GRAPH_PATH = "../files/test_graph.json"

contracts: set[SmartContract] = get_smart_contracts_by_issuer(CONTRACT_ISSUER_ADDRESS)

graph: TransactionsGraph = TransactionsGraphBuilder(contracts, FROM_TIME, TO_TIME).build_graph()
print(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

save_graph_to_json(graph, GRAPH_PATH)