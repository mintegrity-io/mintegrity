from scripts.graph.analysis.metrics.graph_depth import calculate_graph_depth_metrics
from scripts.graph.util.transactions_graph_json import load_graph_from_json

GRAPH_NAME = "rocket_pool_full_graph_90_days"
GRAPH_PATH = f"./files/{GRAPH_NAME}.json"
graph = load_graph_from_json(GRAPH_PATH)

print(calculate_graph_depth_metrics(graph, root_address="0xdd3f50f8a6cafbe9b31a427582963f465e745af8"))
print(calculate_graph_depth_metrics(graph, root_address="0x1d8f8f00cfa6758d7be78336684788fb0ee0fa46"))