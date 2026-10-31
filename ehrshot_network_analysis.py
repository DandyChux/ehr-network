# %%
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cohort import build_cohorts, load_omop_tables
from graph import build_comparison_graph, build_graph
from network_metrics import (
    calculate_metrics,
    calculate_top_diagnosis_overlap,
    permutation_modularity_test,
)
from plot_network import plot_comparison_network, plot_network

# %%

# Load data and construct cohorts
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tables = load_omop_tables(DATA_DIR)

cohorts = build_cohorts(
    tables,
    min_patient_count=10,
)

{
    cohort_name: {
        "rows": len(events),
        "patients": events["person_id"].nunique(),
        "conditions": events["condition_concept_id"].nunique(),
        "deceased_patients": events.loc[events["deceased"], "person_id"].nunique(),
    }
    for cohort_name, events in cohorts.items()
}
# %%

# Build and export graphs
graphs = {}
metrics_by_cohort = {}
metrics_rows = []

for cohort_name, events in cohorts.items():
    graph = build_graph(events, min_pmi=0.0)
    metrics = calculate_metrics(graph)

    graphs[cohort_name] = graph
    metrics_by_cohort[cohort_name] = metrics
    metrics_rows.append(metrics.assign(cohort=cohort_name))
# %%

comparison_graph = build_comparison_graph(
    graphs,
    metrics_by_cohort,
)

# Shared node list: one row per node across all cohorts.
all_node_ids = sorted({str(node) for graph in graphs.values() for node in graph.nodes})

# Map graph node IDs to human-readable OMOP concept metadata.
node_mapping_df = pd.DataFrame({"node_id": all_node_ids})
node_mapping_df["node_type"] = node_mapping_df["node_id"].map(
    lambda node_id: "death" if node_id == "NODE_DEATH" else "condition"
)

concept_columns = [
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
    "concept_code",
]
for column in concept_columns:
    node_mapping_df[column] = pd.NA

concept_path = DATA_DIR / "concept.csv"

if concept_path.exists():
    concept_metadata = pd.read_csv(
        concept_path,
        usecols=["concept_id", *concept_columns],
        dtype={"concept_id": "string"},
    ).rename(columns={"concept_id": "node_id"})

    concept_metadata["node_id"] = concept_metadata["node_id"].astype(str)
    concept_metadata = concept_metadata.drop_duplicates("node_id")

    node_mapping_df = node_mapping_df.drop(columns=concept_columns).merge(
        concept_metadata,
        on="node_id",
        how="left",
    )
    node_mapping_df["mapping_status"] = "not_found_in_concept.csv"
    node_mapping_df.loc[node_mapping_df["concept_name"].notna(), "mapping_status"] = (
        "mapped"
    )
else:
    node_mapping_df["mapping_status"] = "concept_vocabulary_not_provided"

death_mask = node_mapping_df["node_id"].eq("NODE_DEATH")
node_mapping_df.loc[death_mask, "concept_name"] = "Death"
node_mapping_df.loc[death_mask, "mapping_status"] = "special_node"

node_mapping_df.to_csv(
    OUTPUT_DIR / "node_mapping.csv",
    index=False,
)

pd.DataFrame({"node_id": all_node_ids}).to_csv(
    OUTPUT_DIR / "nodes.csv",
    index=False,
)

# Combined edge list: one row per edge per cohort.
all_edge_rows = []

for cohort_name, graph in graphs.items():
    edge_pairs = {
        tuple(sorted((str(source), str(target)))) for source, target in graph.edges()
    }

    for source, target in sorted(edge_pairs):
        all_edge_rows.append(
            {
                "source": source,
                "target": target,
                "cohort": cohort_name,
            }
        )

all_edges_df = pd.DataFrame(
    all_edge_rows,
    columns=["source", "target", "cohort"],
).sort_values(
    ["cohort", "source", "target"],
    ignore_index=True,
)

all_edges_df.to_csv(
    OUTPUT_DIR / "edges_all_cohorts.csv",
    index=False,
)

# Keep separate, traditional edge lists for each cohort.
for cohort_name, graph in graphs.items():
    edge_pairs = {
        tuple(sorted((str(source), str(target)))) for source, target in graph.edges()
    }

    pd.DataFrame(
        sorted(edge_pairs),
        columns=["source", "target"],
    ).to_csv(
        OUTPUT_DIR / f"edges_{cohort_name}.csv",
        index=False,
    )

    # A cohort-specific node list, including isolated nodes.
    cohort_node_ids = sorted(str(node) for node in graph.nodes)

    pd.DataFrame({"node_id": cohort_node_ids}).to_csv(
        OUTPUT_DIR / f"nodes_{cohort_name}.csv",
        index=False,
    )

# Give every GraphML edge a unique key while retaining its cohort attribute.
graphml_export = nx.MultiGraph()
graphml_export.graph.update(comparison_graph.graph)

for node, attributes in comparison_graph.nodes(data=True):
    graphml_export.add_node(node, **attributes)

for edge_number, (source, target, _, attributes) in enumerate(
    comparison_graph.edges(keys=True, data=True)
):
    graphml_export.add_edge(
        source,
        target,
        key=f"edge_{edge_number}",
        **attributes,
    )

nx.write_graphml(
    graphml_export,
    OUTPUT_DIR / "demographic_comparison.graphml",
)

plot_comparison_network(
    graph=comparison_graph,
    output_path=OUTPUT_DIR / "demographic_comparison_network.png",
    title="Female Condition Networks by Demographic Group",
    max_nodes=20,
    max_edges_per_cohort=25,
    min_pmi=0.4,
    min_raw_count=20,
    label_top_n=10,
)

# Useful summary of shared top diagnoses across demographic groups.
overlap_df = calculate_top_diagnosis_overlap(
    metrics_by_cohort,
    top_k=20,
)

overlap_df.to_csv(
    OUTPUT_DIR / "top_diagnosis_jaccard_overlap.csv",
    index=False,
)

print(
    f"Combined graph: {comparison_graph.number_of_nodes()} nodes, "
    f"{comparison_graph.number_of_edges()} cohort-specific edges"
)
# %%

# Generate visualizations
for cohort_name, graph in graphs.items():
    plot_network(
        graph=graph,
        metrics=metrics_by_cohort[cohort_name],
        output_path=OUTPUT_DIR / f"{cohort_name}_network.png",
        title=(f"{cohort_name.replace('_', ' ').title()} Female Condition Network"),
    )
# %%

# Compare top diagnoses
overlap_df = calculate_top_diagnosis_overlap(
    metrics_by_cohort,
    top_k=20,
)

overlap_df.to_csv(
    OUTPUT_DIR / "top_diagnosis_jaccard_overlap.csv",
    index=False,
)

print(overlap_df)
# %%

modularity_test = permutation_modularity_test(
    cohort_events=cohorts,
    graph_builder=lambda events: build_graph(
        events,
        min_pmi=0.4,
    ),
    cohort_a="black_female",
    cohort_b="non_hispanic_white_female",
    permutations=500,
    seed=42,
)

pd.DataFrame([modularity_test]).to_csv(
    OUTPUT_DIR / "modularity_permutation_test.csv",
    index=False,
)

print(modularity_test)
