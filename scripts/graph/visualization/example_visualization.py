import json

from scripts.graph.building.transactions_graph_builder import TransactionsGraphBuilder, MAX_NODES_PER_GRAPH
from scripts.graph.transactions_graph import TransactionsGraph
from scripts.graph.visualization.transaction_graph_visualization import visualize_transactions_graph

graph = TransactionsGraph.from_dict(json.load(open("../files/rocket_pool_graph.json")))
fig = visualize_transactions_graph(graph, "../files/transaction_graph_rocket_pool.html", max_nodes=MAX_NODES_PER_GRAPH)
