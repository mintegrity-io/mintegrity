import plotly.graph_objects as go
import networkx as nx
import random
from typing import Optional, Set, Dict, Any
import pandas as pd
from scripts.graph.transactions_graph import TransactionsGraph, NodeType, Node
from scripts.commons.model import Address, AddressType

# TODO make directed edges


import numpy as np

def visualize_transactions_graph(
        graph: TransactionsGraph,
        filename: Optional[str] = None,
        highlight_address: Optional[str] = None,
        max_nodes: Optional[int] = None,
        layout_iterations: int = 50
) -> go.Figure:
    """
    Creates an interactive visualization of the transactions graph using Plotly.

    Args:
        graph: The TransactionsGraph to visualize
        filename: Path to save the visualization (HTML for interactive, PNG/JPG for static)
        highlight_address: Address to highlight in the visualization
        max_nodes: Maximum number of nodes to include (randomly samples if graph is larger)
        layout_iterations: Number of iterations for the layout algorithm

    Returns:
        A Plotly Figure object
    """
    # Create a NetworkX graph
    G = nx.DiGraph()

    # Handle node sampling if needed
    nodes_to_visualize: Set[Node] = graph.nodes
    if max_nodes and len(graph.nodes) > max_nodes:
        if highlight_address:
            # Keep highlight node if provided
            highlight_node = next((n for n in graph.nodes
                                   if n.address.address.lower() == highlight_address.lower()), None)
            if highlight_node:
                # Sample remaining nodes
                other_nodes = graph.nodes - {highlight_node}
                sample_size = min(max_nodes - 1, len(other_nodes))
                sampled_nodes = set(random.sample(list(other_nodes), sample_size))
                nodes_to_visualize = {highlight_node} | sampled_nodes
            else:
                nodes_to_visualize = set(random.sample(list(graph.nodes), max_nodes))
        else:
            nodes_to_visualize = set(random.sample(list(graph.nodes), max_nodes))

    # Add nodes to NetworkX graph
    for node in nodes_to_visualize:
        G.add_node(
            node.address.address,
            type=node.type.name,
            address_type=node.address.type.name,
            is_highlighted=(node.address.address.lower() == highlight_address.lower()) if highlight_address else False
        )

    # Add edges between nodes in our visualization set
    for edge in graph.edges:
        from_addr = edge.from_node.address.address
        to_addr = edge.to_node.address.address

        if from_addr in G.nodes and to_addr in G.nodes:
            tx_count = len(edge.transactions)
            value = edge.get_total_transactions_value()
            G.add_edge(from_addr, to_addr,
                       weight=tx_count,
                       value=value,
                       transactions=tx_count)

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=0.3, iterations=layout_iterations, seed=42)

    # Define colors and node sizes
    node_colors = {
        "ROOT": "#FFC107",  # Amber
        "WALLET": "#2196F3",  # Blue
        "CONTRACT": "#4CAF50"  # Green
    }

    # Create node dataframes by type
    node_traces = []
    for node_type in NodeType:
        type_nodes = [n for n, attrs in G.nodes(data=True) if attrs['type'] == node_type.name]
        if not type_nodes:
            continue

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_hovertext = []

        for node in type_nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[:8] + "...")

            # Size based on connections
            size = 10 + min(G.degree(node) * 2, 20)  # Cap size
            node_size.append(size)

            # Create detailed hover text
            hover = f"<b>Address:</b> {node}<br>"
            hover += f"<b>Type:</b> {node_type.name}<br>"
            hover += f"<b>Connections:</b> {G.degree(node)}<br>"
            if G.nodes[node].get('is_highlighted', False):
                hover += "<b>HIGHLIGHTED NODE</b>"
            node_hovertext.append(hover)

        # Color marker borders for highlighted node
        marker_line_width = [3 if G.nodes[n].get('is_highlighted', False) else 1 for n in type_nodes]
        marker_line_color = ["red" if G.nodes[n].get('is_highlighted', False) else "white" for n in type_nodes]

        # Create the Plotly trace for this node type
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            name=node_type.name,
            marker=dict(
                color=node_colors[node_type.name],
                size=node_size,
                line=dict(width=marker_line_width, color=marker_line_color)
            ),
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            hovertext=node_hovertext,
            textfont=dict(size=10)
        )
        node_traces.append(node_trace)

    # Create edge traces
    edge_traces = []

    # Group edges by weight for better visualization
    weight_groups = {}
    for edge in G.edges(data=True):
        weight = edge[2]['weight']
        group = min(weight, 10)  # Cap at 10 groups
        if group not in weight_groups:
            weight_groups[group] = []
        weight_groups[group].append(edge)

    # Create a trace for each weight group
    for weight, edges in weight_groups.items():
        edge_x = []
        edge_y = []
        edge_hover = []

        for edge in edges:
            source, target = edge[0], edge[1]
            x0, y0 = pos[source]
            x1, y1 = pos[target]

            # Calculate distance between nodes
            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

            # Create curved path with more points for smoother edges
            # Stronger curve for shorter edges to avoid overlap
            curve_strength = max(0.1, 0.2 - dist * 0.1)

            # Create a bezier curve for the edge path
            n_points = 20
            for i in range(n_points + 1):
                t = i / n_points
                # Calculate bezier curve point
                control_x = (x0 + x1) / 2 + (y1 - y0) * curve_strength
                control_y = (y0 + y1) / 2 - (x1 - x0) * curve_strength
                # Quadratic bezier formula
                bx = (1-t)**2 * x0 + 2*(1-t)*t * control_x + t**2 * x1
                by = (1-t)**2 * y0 + 2*(1-t)*t * control_y + t**2 * y1

                edge_x.append(bx)
                edge_y.append(by)

                # Add hover text to each point
                hover_text = f"<b>From:</b> {source[:8]}...<br>" + \
                             f"<b>To:</b> {target[:8]}...<br>" + \
                             f"<b>Transactions:</b> {edge[2]['transactions']}<br>" + \
                             f"<b>Total Value:</b> {edge[2]['value']:.8f}"
                edge_hover.append(hover_text)

            # Add None to separate edges
            edge_x.append(None)
            edge_y.append(None)
            edge_hover.append(None)

        # Create a trace for this weight group
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=0.5 + (weight * 0.5),
                color='rgba(150, 150, 150, 0.6)'
            ),
            hoverinfo='text',
            hovertext=edge_hover,
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Add arrow markers at the end of each edge
    for edge in G.edges(data=True):
        source, target = edge[0], edge[1]
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        # Calculate edge direction vector
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx*dx + dy*dy)

        # Normalize the vector
        dx, dy = dx/dist, dy/dist

        # Calculate the position for arrow (at 75% of the edge length)
        arrow_x = x0 + dx * dist * 0.75
        arrow_y = y0 + dy * dist * 0.75

        # Calculate size based on edge weight
        weight = edge[2]['weight']
        size = min(8 + weight * 0.5, 15)

        # Add arrow marker
        arrow_trace = go.Scatter(
            x=[arrow_x],
            y=[arrow_y],
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=size,
                color='rgba(80, 80, 80, 0.9)',
                angle=np.arctan2(dy, dx) * 180 / np.pi,
                line=dict(width=1, color='rgba(50, 50, 50, 0.8)')
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(arrow_trace)

    # Create figure
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=f'Transaction Graph - {len(G.nodes)} nodes, {len(G.edges)} edges, {sum(d["transactions"] for _, _, d in G.edges(data=True))} transactions',
            showlegend=True,
            legend=dict(
                title="Node Types",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
    )

    # Add zoom and pan tools
    fig.update_layout(
        dragmode='pan',
        hoverlabel=dict(bgcolor="white", font_size=12),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='Reset',
                     method='relayout',
                     args=[{'xaxis.range': None, 'yaxis.range': None}])
            ],
            x=0.05,
            y=0.05
        )]
    )

    # Save the figure if filename provided
    if filename:
        if filename.endswith('.html'):
            fig.write_html(filename)
        else:
            fig.write_image(filename, width=1200, height=800, scale=2)
        print(f"Graph visualization saved to {filename}")

    return fig

# Usage example:
# graph = TransactionsGraph.from_dict(json.load(open("your_graph.json")))
# fig = visualize_transactions_graph(graph, "transaction_graph.html", max_nodes=100)
# fig.show()  # Display in notebook or browser
