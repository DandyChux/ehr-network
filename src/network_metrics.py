from __future__ import annotations

import networkx as nx
import pandas as pd


def _condition_subgraph(graph: nx.Graph) -> nx.Graph:
    condition_nodes = [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "condition"
    ]

    return graph.subgraph(condition_nodes).copy()


def calculate_metrics(graph: nx.Graph) -> pd.DataFrame:
    """
    Calculate node-level centrality and mortality metrics.
    """

    condition_graph = _condition_subgraph(graph)

    if condition_graph.number_of_nodes() == 0:
        return pd.DataFrame()

    degree_centrality = nx.degree_centrality(condition_graph)

    weighted_degree = dict(condition_graph.degree(weight="weight"))

    betweenness = nx.betweenness_centrality(
        condition_graph,
        weight="distance",
        normalized=True,
    )

    communities = list(
        nx.community.louvain_communities(
            condition_graph,
            weight="weight",
            seed=42,
        )
    )

    community_by_node = {}

    for community_id, community in enumerate(communities):
        for node in community:
            community_by_node[node] = community_id

    modularity = nx.community.modularity(
        condition_graph,
        communities,
        weight="weight",
    )

    rows = []

    for node in condition_graph.nodes:
        death_edge = graph.get_edge_data(node, "NODE_DEATH", {})

        rows.append(
            {
                "node_id": node,
                "node_type": "condition",
                "degree": condition_graph.degree(node),
                "degree_centrality": degree_centrality.get(node, 0.0),
                "weighted_degree": weighted_degree.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "community_id": community_by_node.get(node, -1),
                "case_fatality_ratio": death_edge.get("case_fatality_ratio", 0.0),
                "fatality_count": death_edge.get("fatality_count", 0),
                "modularity": modularity,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "betweenness_centrality",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_top_diagnosis_overlap(
    metrics_by_cohort: dict[str, pd.DataFrame],
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Compute pairwise Jaccard overlap among the top diagnoses.
    """

    top_nodes = {}

    for cohort_name, metrics in metrics_by_cohort.items():
        top_nodes[cohort_name] = set(metrics.head(top_k)["node_id"].astype(str))

    rows = []
    cohort_names = sorted(top_nodes)

    for index, left_name in enumerate(cohort_names):
        for right_name in cohort_names[index + 1 :]:
            left = top_nodes[left_name]
            right = top_nodes[right_name]

            union = left | right
            intersection = left & right

            rows.append(
                {
                    "cohort_a": left_name,
                    "cohort_b": right_name,
                    "top_k": top_k,
                    "intersection_size": len(intersection),
                    "union_size": len(union),
                    "jaccard_index": (len(intersection) / len(union) if union else 0.0),
                }
            )

    return pd.DataFrame(rows)


def calculate_modularity(
    graph: nx.Graph,
) -> float:
    condition_graph = _condition_subgraph(graph)

    if condition_graph.number_of_nodes() < 2:
        return 0.0

    communities = list(
        nx.community.louvain_communities(
            condition_graph,
            weight="weight",
            seed=42,
        )
    )

    return nx.community.modularity(
        condition_graph,
        communities,
        weight="weight",
    )


def permutation_modularity_test(
    cohort_events: dict[str, pd.DataFrame],
    graph_builder,
    cohort_a: str,
    cohort_b: str,
    permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Test the observed modularity difference between two cohorts.

    Patient records are pooled and randomly reassigned to the two
    cohorts while preserving each cohort's original patient count.
    """

    import numpy as np

    if cohort_a not in cohort_events or cohort_b not in cohort_events:
        raise KeyError("Both requested cohorts must exist.")

    left = cohort_events[cohort_a].copy()
    right = cohort_events[cohort_b].copy()

    left["_source_cohort"] = cohort_a
    right["_source_cohort"] = cohort_b

    combined = pd.concat([left, right], ignore_index=True)

    patient_ids = combined["person_id"].drop_duplicates().to_numpy()

    left_patient_ids = set(left["person_id"].unique())
    right_patient_ids = set(right["person_id"].unique())

    if left_patient_ids & right_patient_ids:
        raise ValueError("Permutation testing requires disjoint patient cohorts.")

    observed_left = calculate_modularity(graph_builder(left))
    observed_right = calculate_modularity(graph_builder(right))
    observed_difference = observed_left - observed_right

    rng = np.random.default_rng(seed)
    null_differences = []

    left_patient_count = len(left_patient_ids)

    for _ in range(permutations):
        shuffled = rng.permutation(patient_ids)
        reassigned_left = set(shuffled[:left_patient_count])

        permuted_left = combined[combined["person_id"].isin(reassigned_left)]
        permuted_right = combined[~combined["person_id"].isin(reassigned_left)]

        left_modularity = calculate_modularity(graph_builder(permuted_left))
        right_modularity = calculate_modularity(graph_builder(permuted_right))

        null_differences.append(left_modularity - right_modularity)

    null_differences = np.asarray(null_differences)

    p_value = ((np.abs(null_differences) >= abs(observed_difference)).sum() + 1) / (
        len(null_differences) + 1
    )

    return {
        "cohort_a": cohort_a,
        "cohort_b": cohort_b,
        "observed_modularity_a": observed_left,
        "observed_modularity_b": observed_right,
        "observed_difference": observed_difference,
        "permutations": permutations,
        "p_value": float(p_value),
    }
