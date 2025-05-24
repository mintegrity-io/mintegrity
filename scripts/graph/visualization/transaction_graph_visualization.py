import random
from typing import Optional, Dict, Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from scripts.graph.categorization.graph_categorizer import CategorizedNode
from scripts.graph.model.transactions_graph import TransactionsGraph, NodeType
from scripts.commons.logging_config import get_logger

log = get_logger()


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
    nodes_to_visualize = list(graph.nodes.values())
    if max_nodes and len(graph.nodes) > max_nodes:
        if highlight_address:
            # Keep highlight node if provided
            highlight_node = graph.nodes.get(highlight_address.lower())
            if highlight_node:
                # Sample remaining nodes
                other_nodes = [n for n in nodes_to_visualize if n.address.address.lower() != highlight_address.lower()]
                sample_size = min(max_nodes - 1, len(other_nodes))
                sampled_nodes = random.sample(other_nodes, sample_size)
                nodes_to_visualize = [highlight_node] + sampled_nodes
            else:
                nodes_to_visualize = random.sample(nodes_to_visualize, max_nodes)
        else:
            nodes_to_visualize = random.sample(nodes_to_visualize, max_nodes)

    # Add nodes to NetworkX graph
    for node in nodes_to_visualize:
        G.add_node(
            node.address.address,
            type=node.type.name,
            address_type=node.address.type.name,
            is_highlighted=(node.address.address.lower() == highlight_address.lower()) if highlight_address else False
        )

    # Add edges between nodes in our visualization set
    for edge_key, edge in graph.edges.items():
        from_addr = edge.from_node.address.address
        to_addr = edge.to_node.address.address

        if from_addr in G.nodes and to_addr in G.nodes:
            tx_count = len(edge.transactions)
            value = edge.get_total_transactions_value_usd()
            G.add_edge(from_addr, to_addr,
                       weight=value,  # Using transaction value as weight
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
        # Use logarithmic binning for transaction values
        group = min(int(np.log1p(weight) * 2), 10)  # Cap at 10 groups with log scaling
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

        # Create a trace for this weight group with logarithmic scaling for width
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=0.5 + np.log1p(weight) * 1.0,  # Logarithmic scaling for line width
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

        # Calculate size based on edge weight with logarithmic scaling
        weight = edge[2]['weight']
        size = min(8 + np.log1p(weight) * 0.3, 15)  # Logarithmic scaling for arrow size

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


def visualize_categorized_transactions_graph(
        graph: TransactionsGraph,
        categorized_nodes: Dict[str, CategorizedNode],
        filename: Optional[str] = None,
        highlight_address: Optional[str] = None,
        max_nodes: Optional[int] = None,
        layout_iterations: int = 50
) -> go.Figure:
    """
    Creates an interactive visualization of the transactions graph with categorization data.

    Args:
        graph: The TransactionsGraph to visualize
        categorized_nodes: Dictionary mapping addresses to categorized nodes
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
    nodes_to_visualize = list(graph.nodes.values())
    if max_nodes and len(graph.nodes) > max_nodes:
        if highlight_address:
            # Keep highlight node if provided
            highlight_node = graph.nodes.get(highlight_address.lower())
            if highlight_node:
                # Sample remaining nodes
                other_nodes = [n for n in nodes_to_visualize if n.address.address.lower() != highlight_address.lower()]
                sample_size = min(max_nodes - 1, len(other_nodes))
                sampled_nodes = random.sample(other_nodes, sample_size)
                nodes_to_visualize = [highlight_node] + sampled_nodes
            else:
                nodes_to_visualize = random.sample(nodes_to_visualize, max_nodes)
        else:
            nodes_to_visualize = random.sample(nodes_to_visualize, max_nodes)

    # Add nodes to NetworkX graph with categorization data
    for node in nodes_to_visualize:
        # Get categorization info if available
        node_category = None
        if node.address.address in categorized_nodes:
            node_category = categorized_nodes[node.address.address].type

        G.add_node(
            node.address.address,
            type=node.type.name,
            address_type=node.address.type.name,
            category=node_category.name if node_category else None,
            category_type=type(node_category).__name__ if node_category else None,
            is_highlighted=(node.address.address.lower() == highlight_address.lower()) if highlight_address else False
        )

    # Add edges between nodes in our visualization set (same as original function)
    for edge_key, edge in graph.edges.items():
        from_addr = edge.from_node.address.address
        to_addr = edge.to_node.address.address

        if from_addr in G.nodes and to_addr in G.nodes:
            tx_count = len(edge.transactions)
            value = edge.get_total_transactions_value_usd()
            G.add_edge(from_addr, to_addr,
                       weight=value,
                       value=value,
                       transactions=tx_count)

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=0.3, iterations=layout_iterations, seed=42)

    # Create edge traces (same as original function)
    edge_traces = []

    # Group edges by weight for better visualization
    weight_groups = {}
    for edge in G.edges(data=True):
        weight = edge[2]['weight']
        # Use logarithmic binning for transaction values
        group = min(int(np.log1p(weight) * 2), 10)  # Cap at 10 groups with log scaling
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

        # Create a trace for this weight group with logarithmic scaling for width
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=0.5 + np.log1p(weight) * 1.0,  # Logarithmic scaling for line width
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

        # Calculate size based on edge weight with logarithmic scaling
        weight = edge[2]['weight']
        size = min(8 + np.log1p(weight) * 0.3, 15)  # Logarithmic scaling for arrow size

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

    # Define color mappings for contract types and wallet types
    contract_colors = {
        'DEX': '#FF5722',  # Deep Orange
        'LENDING': '#E91E63',  # Pink
        'NFT_MARKETPLACE': '#9C27B0',  # Purple
        'STAKING': '#673AB7',  # Deep Purple
        'TOKEN': '#3F51B5',  # Indigo
        'BRIDGE': '#795548',  # Brown
        'GENERIC': '#607D8B',  # Blue Grey
    }

    wallet_colors = {
        'EXCHANGE': '#F44336',  # Red
        'WHALE': '#009688',  # Teal
        'DUST_WALLET': '#FFEB3B',  # Yellow
        'BRIDGE_OR_MIXER': '#FF9800',  # Orange
        'USER': '#03A9F4',  # Light Blue
    }

    # Create node traces based on categorization
    node_traces = []

    # Group nodes based on category instead of node type
    category_groups = {}

    for node_address, node_attrs in G.nodes(data=True):
        category = node_attrs.get('category')
        if not category:
            # If no category, use the original node type
            group_key = f"NodeType:{node_attrs['type']}"
        else:
            # If categorized, use category name and type
            category_type = node_attrs.get('category_type')
            group_key = f"{category_type}:{category}"

        if group_key not in category_groups:
            category_groups[group_key] = []

        category_groups[group_key].append(node_address)

    # Create a trace for each category group
    for group_key, nodes in category_groups.items():
        if not nodes:
            continue

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_hovertext = []

        for node in nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[:8] + "...")

            # Size based on connections
            size = 10 + min(G.degree(node) * 2, 20)  # Cap size
            node_size.append(size)

            # Create detailed hover text with category information
            hover = f"<b>Address:</b> {node}<br>"
            hover += f"<b>Type:</b> {G.nodes[node]['type']}<br>"

            if G.nodes[node].get('category'):
                hover += f"<b>Category:</b> {G.nodes[node]['category']}<br>"

            hover += f"<b>Connections:</b> {G.degree(node)}<br>"

            if G.nodes[node].get('is_highlighted', False):
                hover += "<b>HIGHLIGHTED NODE</b>"

            node_hovertext.append(hover)

        # Color marker borders for highlighted node
        marker_line_width = [3 if G.nodes[n].get('is_highlighted', False) else 1 for n in nodes]
        marker_line_color = ["red" if G.nodes[n].get('is_highlighted', False) else "white" for n in nodes]

        # Determine the color based on category
        if group_key.startswith('WalletType:'):
            color = wallet_colors.get(group_key.split(':')[1], '#2196F3')  # Default blue if not found
            name = f"Wallet: {group_key.split(':')[1]}"
        elif group_key.startswith('ContractType:'):
            color = contract_colors.get(group_key.split(':')[1], '#4CAF50')  # Default green if not found
            name = f"Contract: {group_key.split(':')[1]}"
        elif group_key.startswith('NodeType:'):
            # Original node type coloring
            node_type = group_key.split(':')[1]
            color = {
                "ROOT": "#FFC107",  # Amber
                "WALLET": "#2196F3",  # Blue (uncategorized)
                "CONTRACT": "#4CAF50"  # Green (uncategorized)
            }.get(node_type, "#9E9E9E")  # Grey default
            name = f"Uncategorized: {node_type}"
        else:
            color = "#9E9E9E"  # Grey default
            name = group_key

        # Create the Plotly trace for this node group
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            name=name,
            marker=dict(
                color=color,
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

    # Create figure with all the traces
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=f'Transaction Graph with Categories - {len(G.nodes)} nodes, {len(G.edges)} edges',
            showlegend=True,
            legend=dict(
                title="Node Categories",
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


def visualize_graph(
        graph: TransactionsGraph,
        filename: Optional[str] = None,
        node_colors: Optional[Dict[str, str or tuple]] = None,
        node_info: Optional[Dict[str, Dict[str, Any]]] = None,
        title: str = "Wallet Groups Analysis",
        max_nodes: Optional[int] = None,
        layout_iterations: int = 100,
        highlight_direct_connections: bool = True
) -> go.Figure:
    """
    Creates an interactive visualization of wallet groups that are likely controlled by the same operator.
    This function is specifically designed to visualize the results of wallet coordination analysis.

    Args:
        graph: The TransactionsGraph to visualize
        filename: Path to save the visualization (HTML for interactive)
        node_colors: Dictionary mapping addresses to colors (for group visualization)
                     Can be either a color string or a tuple (color, group_num)
        node_info: Dictionary mapping addresses to additional information to display in hover
        title: Title for the visualization
        max_nodes: Maximum number of nodes to include (randomly samples if graph is larger)
        layout_iterations: Number of iterations for the layout algorithm
        highlight_direct_connections: Whether to highlight direct connections between grouped wallets

    Returns:
        A Plotly Figure object
    """
    # Create a NetworkX graph
    G = nx.DiGraph()

    # Process node_colors to extract actual colors and group numbers
    processed_colors = {}
    group_num_map = {}  # Map from group number to color

    if node_colors:
        for addr, color_data in node_colors.items():
            if isinstance(color_data, tuple):
                # New format (color, group_num)
                color, group_num = color_data
                processed_colors[addr] = color
                group_num_map[group_num] = color
            else:
                # Old format (just color)
                processed_colors[addr] = color_data

    # Handle node sampling if needed
    nodes_to_visualize = list(graph.nodes.values())
    if max_nodes and len(graph.nodes) > max_nodes:
        # If we have node_colors, prioritize nodes with colors assigned
        if processed_colors:
            colored_nodes = [n for n in nodes_to_visualize
                           if n.address.address in processed_colors and n.type == NodeType.WALLET]
            remaining_nodes = [n for n in nodes_to_visualize
                             if n.address.address not in processed_colors or n.type != NodeType.WALLET]

            # If we still have too many nodes, sample from each category
            if len(colored_nodes) > max_nodes * 0.8:  # Keep 80% colored, 20% uncolored
                colored_sample_size = int(max_nodes * 0.8)
                colored_nodes = random.sample(colored_nodes, colored_sample_size)

            remaining_sample_size = max_nodes - len(colored_nodes)
            if remaining_sample_size > 0 and remaining_nodes:
                remaining_nodes = random.sample(remaining_nodes, min(remaining_sample_size, len(remaining_nodes)))

            nodes_to_visualize = colored_nodes + remaining_nodes
        else:
            nodes_to_visualize = random.sample(nodes_to_visualize, max_nodes)

    log.info(f"Visualizing {len(nodes_to_visualize)} nodes out of {len(graph.nodes)}")

    # Add nodes to NetworkX graph
    for node in nodes_to_visualize:
        # Determine if node belongs to a group (has color assigned)
        group_id = None
        group_num = None
        if processed_colors and node.address.address in processed_colors:
            color = processed_colors[node.address.address]
            # Check if we have group information from the tuple format
            if node_colors and isinstance(node_colors[node.address.address], tuple):
                _, group_num = node_colors[node.address.address]
                group_id = f"Group {group_num}"
            else:
                group_id = color  # Just use the color as group ID

        # Get additional node info if available
        info = {}
        if node_info and node.address.address in node_info:
            info = node_info[node.address.address]
            # If we have group info from node_info, use it
            if 'group' in info and not group_id:
                group_id = info['group']
                if group_id.startswith('Group '):
                    try:
                        group_num = int(group_id.split(' ')[1])
                    except (ValueError, IndexError):
                        pass

        G.add_node(
            node.address.address,
            type=node.type.name,
            address_type=node.address.type.name,
            group=group_id,
            group_num=group_num,  # Store group number separately
            info=info
        )

    # Add edges between nodes in our visualization set
    strong_edges = set()  # Track edges between nodes in the same group
    for edge_key, edge in graph.edges.items():
        from_addr = edge.from_node.address.address
        to_addr = edge.to_node.address.address

        if from_addr in G.nodes and to_addr in G.nodes:
            tx_count = len(edge.transactions)
            value = edge.get_total_transactions_value_usd()

            is_strong_connection = False
            if highlight_direct_connections and processed_colors:
                # Check if both nodes are in the same group
                if (from_addr in processed_colors and to_addr in processed_colors):
                    from_color = processed_colors[from_addr]
                    to_color = processed_colors[to_addr]
                    if from_color == to_color:
                        is_strong_connection = True
                        strong_edges.add((from_addr, to_addr))

            G.add_edge(from_addr, to_addr,
                       weight=value,
                       value=value,
                       transactions=tx_count,
                       is_strong=is_strong_connection)

    # Use spring layout for positioning with stronger attraction for grouped nodes
    pos = nx.spring_layout(G, k=0.3, iterations=layout_iterations, seed=42, weight='weight')

    # Create edge traces
    edge_traces = []

    # First create regular edges
    regular_edges = [(u, v, d) for u, v, d in G.edges(data=True) if not d.get('is_strong', False)]
    if regular_edges:
        edge_x = []
        edge_y = []
        edge_hover = []

        for u, v, d in regular_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Calculate distance between nodes
            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

            # Minimal curve for regular edges
            curve_strength = max(0.05, 0.1 - dist * 0.05)

            # Create a bezier curve for the edge path
            n_points = 15
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
                hover_text = f"<b>From:</b> {u[:8]}...<br>" + \
                           f"<b>To:</b> {v[:8]}...<br>" + \
                           f"<b>Transactions:</b> {d['transactions']}<br>" + \
                           f"<b>Value:</b> ${d['value']:.2f}"
                edge_hover.append(hover_text)

            # Add None to separate edges
            edge_x.append(None)
            edge_y.append(None)
            edge_hover.append(None)

        regular_edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=0.5,
                color='rgba(150, 150, 150, 0.4)'  # Light gray, transparent
            ),
            hoverinfo='text',
            hovertext=edge_hover,
            showlegend=False
        )
        edge_traces.append(regular_edge_trace)

    # Then create "strong" edges (between wallets in same group)
    strong_edges_list = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('is_strong', False)]
    if strong_edges_list:
        edge_x = []
        edge_y = []
        edge_hover = []

        for u, v, d in strong_edges_list:
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Calculate distance between nodes
            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

            # More pronounced curve for strong edges
            curve_strength = max(0.15, 0.2 - dist * 0.05)

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
                hover_text = f"<b>SAME OPERATOR CONNECTION</b><br>" + \
                           f"<b>From:</b> {u[:8]}...<br>" + \
                           f"<b>To:</b> {v[:8]}...<br>" + \
                           f"<b>Transactions:</b> {d['transactions']}<br>" + \
                           f"<b>Value:</b> ${d['value']:.2f}"
                edge_hover.append(hover_text)

            # Add None to separate edges
            edge_x.append(None)
            edge_y.append(None)
            edge_hover.append(None)

        strong_edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=2.0,
                color='rgba(50, 50, 200, 0.7)'  # Bold blue for strong connections
            ),
            hoverinfo='text',
            hovertext=edge_hover,
            name='Same Operator Connection',
            showlegend=True
        )
        edge_traces.append(strong_edge_trace)

    # Create node traces
    node_traces = []

    # Group nodes by their assigned group
    grouped_nodes = {}
    for node, attrs in G.nodes(data=True):
        group = attrs.get('group', None)
        if group not in grouped_nodes:
            grouped_nodes[group] = []
        grouped_nodes[group].append(node)

    # Create color assignment for groups that don't have predefined colors
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Extract group identifiers and ensure proper sorting
    # For "Group X" format, sort numerically, otherwise alphabetically
    group_ids = list(grouped_nodes.keys())

    # Separate numeric groups and other groups
    numeric_groups = []
    other_groups = []

    for group_id in group_ids:
        if group_id is None:
            continue  # Skip None (will be processed separately)
        elif isinstance(group_id, str) and group_id.startswith("Group "):
            try:
                group_num = int(group_id.split(" ")[1])
                numeric_groups.append((group_id, group_num))
            except (ValueError, IndexError):
                other_groups.append(group_id)
        else:
            other_groups.append(group_id)

    # Sort numeric groups by their number
    numeric_groups.sort(key=lambda x: x[1])
    sorted_numeric_groups = [g[0] for g in numeric_groups]

    # Sort other groups alphabetically
    other_groups.sort()

    # Combine sorted lists, with numeric groups first, then other groups, and None at the end
    sorted_group_ids = sorted_numeric_groups + other_groups
    if None in group_ids:
        sorted_group_ids.append(None)

    # Create a trace for each group (in order)
    node_trace_data = []  # We'll collect trace data and sort later

    for group_id in sorted_group_ids:
        nodes = grouped_nodes[group_id]
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_hovertext = []

        for node in nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Shorter display text
            node_text.append(node[:6] + "...")

            # Size based on connections
            size = 10 + min(G.degree(node) * 2, 20)
            node_size.append(size)

            # Create detailed hover text
            hover = f"<b>Address:</b> {node}<br>"
            hover += f"<b>Type:</b> {G.nodes[node]['type']}<br>"
            hover += f"<b>Connections:</b> {G.degree(node)}<br>"

            # Add info from node_info if available
            if node_info and node in node_info:
                info = node_info[node]
                for key, value in info.items():
                    if key != 'address' and key != 'group':  # Skip address since we already show it
                        hover += f"<b>{key.replace('_', ' ').title()}:</b> {value}<br>"

            # Add group information
            if group_id:
                hover += f"<b>Group:</b> {group_id}<br>"

            node_hovertext.append(hover)

        # Determine color and name for the group
        if group_id is None:
            # Ungrouped nodes are gray
            color = "#cccccc"  # Light gray
            name = "Ungrouped Nodes"
            sort_key = "ZZZ"  # Ensures it appears last in legend
        elif isinstance(group_id, str) and group_id.startswith("Group "):
            # For groups with explicit numbering
            try:
                group_num = int(group_id.split(" ")[1])
                name = f"Group {group_num}"

                # Use predefined color from group_num_map if available
                if group_num in group_num_map:
                    color = group_num_map[group_num]
                else:
                    color = base_colors[(group_num - 1) % len(base_colors)]

                # Sort key ensures proper ordering (Group 1, Group 2, etc.)
                sort_key = f"A{group_num:04d}"
            except (ValueError, IndexError):
                # Fallback for malformed "Group X" strings
                color = base_colors[sorted_group_ids.index(group_id) % len(base_colors)]
                name = group_id
                sort_key = f"B{group_id}"
        else:
            # For other group identifiers
            color = base_colors[sorted_group_ids.index(group_id) % len(base_colors)]
            name = str(group_id)
            sort_key = f"C{name}"

        # Create the scatter trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            name=name,
            marker=dict(
                color=color,
                size=node_size,
                line=dict(width=1, color='white')
            ),
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            hovertext=node_hovertext,
            textfont=dict(size=10)
        )

        # Store the trace data with its sort key for later ordering
        node_trace_data.append((sort_key, node_trace))

    # Sort the node traces to ensure consistent legend ordering
    node_trace_data.sort(key=lambda x: x[0])

    # Create the node traces in the sorted order
    sorted_node_traces = [trace for _, trace in node_trace_data]

    # Make sure we don't have any duplicate entries in the legend
    seen_names = set()
    for trace in sorted_node_traces:
        if trace.name in seen_names:
            # If we've already seen this name, don't show it again in the legend
            trace.showlegend = False
        else:
            seen_names.add(trace.name)
            trace.showlegend = True

    # Create figure with all traces - edges first, then node traces
    fig = go.Figure(
        data=edge_traces + sorted_node_traces,
        layout=go.Layout(
            title=title,
            showlegend=True,
            legend=dict(
                title="Wallet Groups",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)',
                # Sort items in the legend
                traceorder='grouped'
            ),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            annotations=[
                dict(
                    text="Wallet coordination analysis: colored groups represent wallets likely managed by the same operator",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    font=dict(size=12)
                )
            ]
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
