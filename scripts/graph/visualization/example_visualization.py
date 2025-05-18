import json

from scripts.commons import initialize_metadata
from scripts.graph.building.transactions_graph_builder import MAX_NODES_PER_GRAPH
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.visualization.transaction_graph_visualization import visualize_transactions_graph

initialize_metadata.init()

GRAPH_FILE_NAME = "rocket_pool_graph_10_days"

graph = TransactionsGraph.from_dict(json.load(open(f"../files/{GRAPH_FILE_NAME}.json")))
fig = visualize_transactions_graph(graph, f"../files/transaction_graph_{GRAPH_FILE_NAME}.html", max_nodes=MAX_NODES_PER_GRAPH)
