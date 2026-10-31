from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def plot_network(
    graph: nx.Graph,
    metrics: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_by_node = metrics.set_index("node_id").to_dict(orient="index")

    # Fruchterman-Reingold layout.
    positions = nx.spring_layout(
        graph,
        seed=42,
        weight="weight",
        iterations=200,
    )

    condition_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "condition"
    ]

    death_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "death"
    ]

    condition_sizes = [
        500
        + 10000
        * metric_by_node.get(node, {}).get(
            "betweenness_centrality",
            0.0,
        )
        for node in condition_nodes
    ]

    condition_colors = [
        metric_by_node.get(node, {}).get("community_id", -1) for node in condition_nodes
    ]

    co_edges = [
        (source, target)
        for source, target, attributes in graph.edges(data=True)
        if attributes.get("edge_type") == "condition_co_occurrence"
    ]

    mortality_edges = [
        (source, target)
        for source, target, attributes in graph.edges(data=True)
        if attributes.get("edge_type") == "condition_mortality"
    ]

    co_widths = [
        max(0.5, graph[source][target].get("pmi", 0.0)) for source, target in co_edges
    ]

    mortality_widths = [
        max(
            0.5,
            5.0
            * graph[source][target].get(
                "case_fatality_ratio",
                0.0,
            ),
        )
        for source, target in mortality_edges
    ]

    plt.figure(figsize=(14, 10), dpi=300)

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=co_edges,
        width=co_widths,
        alpha=0.35,
        edge_color="#64748b",
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=mortality_edges,
        width=mortality_widths,
        alpha=0.8,
        edge_color="#dc2626",
        style="dashed",
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=condition_nodes,
        node_size=condition_sizes,
        node_color=condition_colors,
        cmap="tab20",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
    )

    if death_nodes:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=death_nodes,
            node_size=1800,
            node_color="#dc2626",
            node_shape="D",
            edgecolors="black",
            linewidths=1.0,
        )

    labels = {
        node: "Death" if node == "NODE_DEATH" else str(node) for node in graph.nodes
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=7,
        font_weight="bold",
    )

    plt.title(title, fontsize=15, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _filter_comparison_graph(
    graph: nx.MultiGraph,
    max_nodes: int,
    max_edges_per_cohort: int,
    min_pmi: float,
    min_raw_count: int,
) -> nx.MultiGraph:
    """Create a smaller graph for plotting; leave the source graph unchanged."""
    condition_nodes = [
        (node, attributes)
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "condition"
    ]

    def node_score(attributes: dict) -> float:
        values = [
            float(value)
            for name, value in attributes.items()
            if name.endswith("__betweenness_centrality") and value is not None
        ]
        return max(values, default=0.0)

    ranked_nodes = sorted(
        condition_nodes,
        key=lambda item: node_score(item[1]),
        reverse=True,
    )[:max_nodes]

    selected_nodes = {node for node, _ in ranked_nodes}
    selected_nodes.update(
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "death"
    )

    plot_graph = nx.MultiGraph()
    plot_graph.graph.update(graph.graph)

    for node in selected_nodes:
        plot_graph.add_node(node, **graph.nodes[node])

    edges_by_cohort = defaultdict(list)

    for source, target, key, attributes in graph.edges(
        keys=True,
        data=True,
    ):
        if source not in selected_nodes or target not in selected_nodes:
            continue

        edge_type = attributes.get("edge_type")
        raw_count = int(attributes.get("raw_count", 0))

        if edge_type == "condition_co_occurrence":
            pmi = float(attributes.get("pmi", 0.0))
            if raw_count < min_raw_count or pmi < min_pmi:
                continue
            score = pmi
        else:
            # Keep mortality edges, but rank them by their fatality rate.
            score = float(attributes.get("case_fatality_ratio", 0.0))

        cohort = attributes.get("cohort", key)
        edges_by_cohort[cohort].append(
            (score, raw_count, source, target, dict(attributes))
        )

    for cohort, edges in edges_by_cohort.items():
        edges.sort(key=lambda item: (item[0], item[1]), reverse=True)

        for index, (_, _, source, target, attributes) in enumerate(
            edges[:max_edges_per_cohort]
        ):
            plot_graph.add_edge(
                source,
                target,
                key=f"{cohort}_{index}",
                **attributes,
            )

    return plot_graph


def plot_comparison_network(
    graph: nx.MultiGraph,
    output_path: str | Path,
    title: str,
    max_nodes: int = 35,
    max_edges_per_cohort: int = 40,
    min_pmi: float = 0.25,
    min_raw_count: int = 5,
    label_top_n: int = 12,
) -> None:
    """
    Plot a demographic comparison MultiGraph.

    Edge color indicates cohort. Parallel edges between the same
    conditions are curved apart to keep them visible.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = _filter_comparison_graph(
        graph,
        max_nodes=max_nodes,
        max_edges_per_cohort=max_edges_per_cohort,
        min_pmi=min_pmi,
        min_raw_count=min_raw_count,
    )

    cohorts = sorted(
        {
            attributes.get("cohort", key)
            for _, _, key, attributes in graph.edges(
                keys=True,
                data=True,
            )
        }
    )

    palette = plt.get_cmap("tab10")
    cohort_colors = {
        cohort: palette(index % 10) for index, cohort in enumerate(cohorts)
    }

    # Use a simple graph for layout. Co-occurrence PMI informs
    # condition spacing; mortality links get a weaker layout weight.
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(graph.nodes)

    for source, target, _, attributes in graph.edges(
        keys=True,
        data=True,
    ):
        edge_type = attributes.get("edge_type")

        if edge_type == "condition_co_occurrence":
            layout_weight = max(
                float(attributes.get("pmi", 0.0)),
                0.01,
            )
        else:
            layout_weight = 0.1

        if layout_graph.has_edge(source, target):
            layout_graph[source][target]["weight"] += layout_weight
        else:
            layout_graph.add_edge(
                source,
                target,
                weight=layout_weight,
            )

    positions = nx.spring_layout(
        layout_graph,
        seed=42,
        weight="weight",
        iterations=250,
    )

    # Group edges by node pair so subgroup-specific edges can be curved
    # in opposite directions instead of being drawn on top of each other.
    edges_by_pair = defaultdict(list)

    for source, target, key, attributes in graph.edges(
        keys=True,
        data=True,
    ):
        pair = tuple(sorted((str(source), str(target))))
        edges_by_pair[pair].append((source, target, key, attributes))

    fig, ax = plt.subplots(figsize=(15, 11), dpi=300)

    for pair_edges in edges_by_pair.values():
        pair_edges.sort(key=lambda edge: edge[3].get("cohort", edge[2]))

        edge_count = len(pair_edges)

        for index, (source, target, key, attributes) in enumerate(pair_edges):
            cohort = attributes.get("cohort", key)
            edge_type = attributes.get("edge_type")

            # A single edge stays straight. Multiple group edges curve
            # symmetrically around the same node-to-node connection.
            radius = 0.0 if edge_count == 1 else (index - (edge_count - 1) / 2) * 0.18

            pmi = max(float(attributes.get("pmi", 0.0)), 0.0)
            width = min(4.5, max(0.8, 1.0 + pmi))

            patch = FancyArrowPatch(
                positions[source],
                positions[target],
                connectionstyle=f"arc3,rad={radius}",
                arrowstyle="-",
                linewidth=width,
                linestyle=("--" if edge_type == "condition_mortality" else "-"),
                color=cohort_colors[cohort],
                alpha=0.75,
                zorder=1,
            )
            ax.add_patch(patch)

    condition_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "condition"
    ]

    death_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "death"
    ]

    # Size nodes by the largest betweenness value across cohorts.
    condition_sizes = []
    for node in condition_nodes:
        betweenness_values = [
            float(value)
            for attribute, value in graph.nodes[node].items()
            if attribute.endswith("__betweenness_centrality") and value is not None
        ]
        max_betweenness = max(betweenness_values, default=0.0)
        condition_sizes.append(400 + 10000 * max_betweenness)

    condition_artist = nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=condition_nodes,
        node_size=condition_sizes,
        node_color="#cbd5e1",
        edgecolors="#334155",
        linewidths=0.6,
        ax=ax,
    )
    condition_artist.set_zorder(2)

    if death_nodes:
        death_artist = nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=death_nodes,
            node_size=1500,
            node_color="#fecaca",
            node_shape="D",
            edgecolors="#991b1b",
            linewidths=1.0,
            ax=ax,
        )
        death_artist.set_zorder(3)

    def node_betweeness(node: str) -> float:
        values = [
            float(value)
            for name, value in graph.nodes[node].items()
            if name.endswith("__betweenness_centrality") and value is not None
        ]
        return max(values, default=0.0)

    top_labels = set(
        sorted(
            condition_nodes,
            key=node_betweeness,
            reverse=True,
        )[:label_top_n]
    )

    labels = {
        node: (
            "Death"
            if graph.nodes[node].get("node_type") == "death"
            else str(graph.nodes[node].get("label", node))
        )
        for node in graph.nodes
        if node in top_labels or graph.nodes[node].get("node_type") == "death"
    }

    label_artists = nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=7,
        font_weight="bold",
        ax=ax,
    )

    for label_artist in label_artists.values():
        label_artist.set_zorder(4)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=cohort_colors[cohort],
            linewidth=2,
            label=cohort.replace("_", " ").title(),
        )
        for cohort in cohorts
    ]

    ax.legend(
        handles=legend_handles,
        title="Demographic group",
        loc="best",
        frameon=True,
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
