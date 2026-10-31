from __future__ import annotations

from itertools import combinations
from math import log

import networkx as nx
import pandas as pd

NODE_DEATH = "NODE_DEATH"


def _safe_pmi(
    pair_count: int,
    left_count: int,
    right_count: int,
    total_events: int,
) -> float:
    if min(pair_count, left_count, right_count, total_events) <= 0:
        return 0.0

    probability_pair = pair_count / total_events
    probability_left = left_count / total_events
    probability_right = right_count / total_events

    return log(probability_pair / (probability_left * probability_right))


def build_graph(
    cohort_events: pd.DataFrame,
    min_pmi: float | None = None,
) -> nx.Graph:
    """
    Build an undirected condition co-occurrence graph.

    Conditions are co-occurrence nodes. NODE_DEATH is linked to
    every condition observed for a patient who appears in death.csv.
    """

    required_columns = {
        "person_id",
        "visit_occurrence_id",
        "condition_concept_id",
        "deceased",
    }

    missing = required_columns - set(cohort_events.columns)
    if missing:
        raise ValueError(f"Missing graph input columns: {sorted(missing)}")

    events = (
        cohort_events[
            [
                "person_id",
                "visit_occurrence_id",
                "condition_concept_id",
                "deceased",
            ]
        ]
        .dropna()
        .copy()
    )

    events["condition_concept_id"] = events["condition_concept_id"].astype(str)

    graph = nx.Graph()

    # One condition should be counted once per patient visit.
    visit_conditions = events.groupby(
        ["person_id", "visit_occurrence_id"],
        as_index=False,
    )["condition_concept_id"].agg(lambda values: sorted(set(values)))

    total_visits = len(visit_conditions)

    if total_visits == 0:
        return graph

    condition_visit_counts: dict[str, int] = {}

    for conditions in visit_conditions["condition_concept_id"]:
        for condition in conditions:
            condition_visit_counts[condition] = (
                condition_visit_counts.get(condition, 0) + 1
            )

    for condition, visit_count in condition_visit_counts.items():
        graph.add_node(
            condition,
            node_type="condition",
            condition_concept_id=condition,
            visit_count=visit_count,
        )

    pair_counts: dict[tuple[str, str], int] = {}

    for conditions in visit_conditions["condition_concept_id"]:
        for left, right in combinations(conditions, 2):
            pair = tuple(sorted((left, right)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    for (left, right), pair_count in pair_counts.items():
        pmi = _safe_pmi(
            pair_count=pair_count,
            left_count=condition_visit_counts[left],
            right_count=condition_visit_counts[right],
            total_events=total_visits,
        )

        if min_pmi is not None and pmi < min_pmi:
            continue

        graph.add_edge(
            left,
            right,
            edge_type="condition_co_occurrence",
            raw_count=pair_count,
            pmi=pmi,
            weight=max(pmi, 0.0),
            distance=1.0 / max(pmi, 1e-12),
        )

    # Add the terminal mortality node only when the cohort has deaths.
    deceased_events = events[events["deceased"]]

    if not deceased_events.empty:
        graph.add_node(
            NODE_DEATH,
            node_type="death",
            label="Death",
        )

        deceased_patients = set(deceased_events["person_id"].unique())

        patient_condition_counts = events.groupby("condition_concept_id")[
            "person_id"
        ].nunique()

        deceased_condition_counts = deceased_events.groupby("condition_concept_id")[
            "person_id"
        ].nunique()

        for condition in deceased_condition_counts.index:
            deaths = int(deceased_condition_counts[condition])
            patients = int(patient_condition_counts[condition])

            graph.add_edge(
                condition,
                NODE_DEATH,
                edge_type="condition_mortality",
                raw_count=deaths,
                fatality_count=deaths,
                case_fatality_ratio=deaths / patients,
                weight=deaths / patients,
                distance=1.0 / max(deaths / patients, 1e-12),
            )

    graph.graph["total_visits"] = total_visits
    graph.graph["deceased_patient_count"] = (
        len(deceased_patients) if (not deceased_events.empty) else 0
    )

    return graph


def graph_to_tables(
    graph: nx.Graph,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = []

    for node, attributes in graph.nodes(data=True):
        row = {"node_id": node}
        row.update(attributes)
        nodes.append(row)

    edges = []

    for source, target, attributes in graph.edges(data=True):
        row = {
            "source_node": source,
            "target_node": target,
        }
        row.update(attributes)
        edges.append(row)

    return pd.DataFrame(nodes), pd.DataFrame(edges)


def build_comparison_graph(
    graphs: dict[str, nx.Graph],
    metrics_by_cohort: dict[str, pd.DataFrame],
) -> nx.MultiGraph:
    """
    Combine cohort graphs while preserving each cohort's edge attributes.

    There is one node per condition concept. A node pair can have
    multiple edges, one keyed by cohort.
    """
    comparison = nx.MultiGraph()
    comparison.graph["graph_type"] = "demographic_comparison"

    metric_columns = [
        "degree",
        "degree_centrality",
        "weighted_degree",
        "betweenness_centrality",
        "community_id",
        "case_fatality_ratio",
        "fatality_count",
        "modularity",
    ]

    for cohort_name, graph in graphs.items():
        metrics = metrics_by_cohort.get(cohort_name, pd.DataFrame())

        if not metrics.empty and "node_id" in metrics.columns:
            metrics_by_node = metrics.set_index("node_id").to_dict(orient="index")
        else:
            metrics_by_node = {}

        for node, attributes in graph.nodes(data=True):
            if node not in comparison:
                comparison.add_node(node, **attributes)

            node_metrics = metrics_by_node.get(node, {})
            for column in metric_columns:
                if column in node_metrics:
                    comparison.nodes[node][f"{cohort_name}__{column}"] = node_metrics[
                        column
                    ]

        for source, target, attributes in graph.edges(data=True):
            comparison.add_edge(
                source,
                target,
                key=cohort_name,
                cohort=cohort_name,
                **attributes,
            )

    return comparison


def comparison_graph_to_tables(
    graph: nx.MultiGraph,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export one row per node and one row per cohort-specific edge.
    """
    nodes = []

    for node, attributes in graph.nodes(data=True):
        nodes.append({"node_id": node, **attributes})

    edges = []

    for source, target, key, attributes in graph.edges(
        keys=True,
        data=True,
    ):
        edges.append(
            {
                "source_node": source,
                "target_node": target,
                "cohort": attributes.get("cohort", key),
                **attributes,
            }
        )

    return pd.DataFrame(nodes), pd.DataFrame(edges)
