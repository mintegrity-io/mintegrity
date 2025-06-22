import json

from scripts.commons import prices
from scripts.graph.building.transactions_graph_builder import MAX_NODES_PER_GRAPH_DEFAULT
from scripts.graph.categorization.graph_categorizer import CategorizedNode, categorize_graph
from scripts.graph.model.transactions_graph import TransactionsGraph
from scripts.graph.visualization.transaction_graph_visualization import visualize_transactions_graph, visualize_categorized_transactions_graph

prices.init()

GRAPH_NAME = "rocket_pool_full_graph_90_days"
GRAPH_PATH = f"./files/{GRAPH_NAME}.json"
graph = TransactionsGraph.from_dict(json.load(open(GRAPH_PATH)))

visualize_transactions_graph(
    graph=graph,
    filename=f"./files/visualized/transaction_graph_{GRAPH_NAME}.html",
    max_nodes=MAX_NODES_PER_GRAPH_DEFAULT)

visualize_transactions_graph(
    graph=graph,
    filename=f"./files/visualized/transaction_graph_{GRAPH_NAME}_optimized.html",
    max_nodes=MAX_NODES_PER_GRAPH_DEFAULT,
    optimize_graph=True)

categorized_nodes: dict[str, CategorizedNode] = categorize_graph(graph)

visualize_categorized_transactions_graph(
    graph=graph,
    categorized_nodes=categorized_nodes,
    filename=f"./files/visualized/transaction_graph_{GRAPH_NAME}_categorized.html",
    max_nodes=MAX_NODES_PER_GRAPH_DEFAULT
)

visualize_categorized_transactions_graph(
    graph=graph,
    categorized_nodes=categorized_nodes,
    filename=f"./files/visualized/transaction_graph_{GRAPH_NAME}_categorized_optimized.html",
    max_nodes=MAX_NODES_PER_GRAPH_DEFAULT,
    optimize_graph=True
)