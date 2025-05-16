import json

from scripts.graph.transactions_graph import TransactionsGraph
from scripts.graph.visualization.transaction_graph_visualization import visualize_transactions_graph

graph = TransactionsGraph.from_dict(json.load(open("../files/test_graph.json")))
fig = visualize_transactions_graph(graph, "transaction_graph.html", max_nodes=100)
fig.show()
