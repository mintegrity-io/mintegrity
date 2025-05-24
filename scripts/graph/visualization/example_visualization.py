import json

from scripts.commons import metadata
from scripts.graph.building.transactions_graph_builder import MAX_NODES_PER_GRAPH
from scripts.graph.categorization.graph_categorizer import CategorizedNode, categorize_graph
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.visualization.transaction_graph_visualization import visualize_transactions_graph, visualize_categorized_transactions_graph

metadata.init()

GRAPH_FILE_NAME = "rocket_pool_graph_10_days"

graph = TransactionsGraph.from_dict(json.load(open(f"../files/{GRAPH_FILE_NAME}.json")))

visualize_transactions_graph(
    graph=graph,
    filename=f"../files/transaction_graph_{GRAPH_FILE_NAME}.html",
    max_nodes=MAX_NODES_PER_GRAPH)

categorized_nodes: dict[str, CategorizedNode] = categorize_graph(graph)

visualize_categorized_transactions_graph(
    graph=graph,
    categorized_nodes=categorized_nodes,
    filename=f"../files/transaction_graph_{GRAPH_FILE_NAME}_categorized.html",
    max_nodes=MAX_NODES_PER_GRAPH
)

