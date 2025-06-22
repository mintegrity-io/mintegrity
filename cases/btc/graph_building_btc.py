import datetime

from dateutil import parser

from scripts.commons import prices
from scripts.commons.model import *
from scripts.graph.building.transactions_graph_builder import TransactionsGraphBuilder, TargetNetwork
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.util.transactions_graph_json import save_graph_to_json

days = 60

TO_TIME = int(datetime.datetime.now().timestamp()) # now should be used with mempool, otherwise it will be very slow
FROM_TIME = TO_TIME - (days * 24 * 60 * 60)

GRAPH_NAME = f"btc_full_graph_{days}_test_days"
GRAPH_PATH = f"./files/{GRAPH_NAME}.json"

prices.init()

root_wallets: set[Address] = {Address("1Ay8vMC7R1UbyCCZRVULMV7iQpHSAbguJP", type=AddressType.WALLET)} # Mr 100
# root_wallets: set[Address] = {Address("3JcWcMPtxGaCuhKeucTMe1V865t7UhrNnT", type=AddressType.WALLET)}

graph: TransactionsGraph = TransactionsGraphBuilder(root_wallets, FROM_TIME, TO_TIME, TargetNetwork.BTC, max_transactions_normal_node=100).build_graph()
print(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

save_graph_to_json(graph, GRAPH_PATH)
