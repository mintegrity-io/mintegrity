"""
Example script for detecting coordinated wallet groups in the Rocket Pool dataset.
This helps identify wallet groups that are likely managed by the same operator,
particularly focusing on layer + storage wallet patterns.
"""
from scripts.graph.analysis.wallet_groups.wallet_grouping import analyze_and_visualize_wallet_groups
from scripts.graph.util.transactions_graph_json import load_graph_from_json

from scripts.commons.logging_config import get_logger

log = get_logger()

# Path to graph file (60 days of data for comprehensive analysis)
GRAPH_NAME = "rocket_pool_full_graph_60_days"
GRAPH_PATH = f"../../files/{GRAPH_NAME}.json"

# Analysis threshold - minimum coordination score to consider wallets related (0-10 scale)
COORDINATION_THRESHOLD = 5

# Path to save coordination visualization
OUTPUT_PATH = f"../../files/wallet_groups/{GRAPH_NAME}_wallet_coordination_groups_{int(COORDINATION_THRESHOLD)}.html"


log.info(f"Loading graph from {GRAPH_PATH}")
graph = load_graph_from_json(GRAPH_PATH)
log.info(f"Graph loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

# Method 1: Full analysis with visualization
log.info("Performing full wallet coordination analysis with visualization...")
wallet_groups, coordination_scores, group_distances = analyze_and_visualize_wallet_groups(graph, OUTPUT_PATH, COORDINATION_THRESHOLD)

# Method 2: Step-by-step analysis (uncomment if needed)
# coordination_scores = detect_wallet_coordination(graph)
# wallet_groups = identify_wallet_groups(coordination_scores, threshold=COORDINATION_THRESHOLD)

# Print summary of results
log.info(f"\n\n{'=' * 50}")
log.info("WALLET GROUP ANALYSIS RESULTS")
log.info(f"{'=' * 50}")
log.info(f"Found {len(wallet_groups)} coordinated wallet groups")

for i, group in enumerate(wallet_groups):
    group_num = i + 1
    distance = group_distances.get(group_num, "unknown")
    distance_info = f"distance to root: {distance}" if distance >= 0 else "no path to root found"
    log.info(f"\nGROUP {group_num}: {len(group)} wallets ({distance_info})")
    log.info("-" * 40)

    # Display each wallet in the group with its coordination scores
    for wallet_addr in group:
        # Get top coordination partners for this wallet
        partners = [(partner, score) for partner, score in coordination_scores.get(wallet_addr, {}).items()
                    if partner in group and partner != wallet_addr]
        partners.sort(key=lambda x: x[1], reverse=True)

        # Display wallet with its top coordination partners
        if partners:
            top_partner, top_score = partners[0]
            top_partner_short = f"{top_partner[:6]}...{top_partner[-4:]}"
            log.info(f"Wallet {wallet_addr[:6]}...{wallet_addr[-4:]} - strongest coordination with {top_partner_short} (score: {top_score:.1f})")
        else:
            log.info(f"Wallet {wallet_addr[:6]}...{wallet_addr[-4:]}")

log.info(f"\n{'=' * 50}")
log.info(f"Visualization saved to: {OUTPUT_PATH}")
log.info(f"{'=' * 50}\n")
