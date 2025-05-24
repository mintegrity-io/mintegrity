import json

from scripts.commons import metadata
from scripts.graph.categorization.graph_categorizer import categorize_graph, CategorizedNode
from scripts.graph.model.transactions_graph import TransactionsGraph

metadata.init()

GRAPH_FILE_NAME = "rocket_pool_graph_10_days"

graph: TransactionsGraph = TransactionsGraph.from_dict(json.load(open(f"../files/{GRAPH_FILE_NAME}.json")))
categorized_nodes: dict[str, CategorizedNode] = categorize_graph(graph)

print(categorized_nodes)