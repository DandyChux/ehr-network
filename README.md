# EHRShot Women's Health Network Analysis

This project builds and compares condition co-occurrence networks for female patient cohorts using OMOP Common Data Model (OMOP-CDM) tables. It analyzes network structure, condition centrality, and the relationship between conditions and recorded mortality.

The workflow is implemented in Python and can be run from the main script or explored step by step in the Jupyter notebook.

> **Research use only:** This project is an analytical workflow, not a clinical decision-support tool. Ensure that data use, storage, and sharing comply with applicable privacy requirements.

## Project structure

```text
ehrshot-network/
├── data/
│   ├── person.csv
│   ├── condition_occurrence.csv
│   ├── visit_occurrence.csv
│   └── death.csv
├── outputs/
│   ├── nodes*.csv
│   ├── edges*.csv
│   ├── *_network.png
│   ├── demographic_comparison.graphml
│   └── top_diagnosis_jaccard_overlap.csv
├── src/
│   ├── cohort.py
│   ├── graph.py
│   ├── network_metrics.py
│   └── plot_network.py
├── EHRShot_Network_Analysis.ipynb
├── ehrshot_network_analysis.py
├── requirements.txt
└── Pipfile
```

## Requirements

- Python 3.10 or later
- Dependencies listed in `requirements.txt`

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Input data

Place the following OMOP-CDM CSV files in `data/`:

| File                       | Purpose                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `person.csv`               | Patient gender, race, and ethnicity concepts                    |
| `condition_occurrence.csv` | Recorded condition concepts and visit identifiers               |
| `visit_occurrence.csv`     | Patient-to-visit associations used to validate condition visits |
| `death.csv`                | Patient identifiers used to determine recorded mortality status |

The tables must include the columns used by the pipeline, including `person_id`, `gender_concept_id`, `race_concept_id`, `ethnicity_concept_id`, `visit_occurrence_id`, and `condition_concept_id`. The cohort module checks required columns and raises an error if any are missing.

## Running the analysis

Run the complete pipeline from the project root:

```bash
python ehrshot_network_analysis.py
```

The script reads input files from `data/`, creates `outputs/` if needed, and writes graph exports, comparison results, and visualizations there.

For an interactive, step-by-step version of the workflow, open `EHRShot_Network_Analysis.ipynb` in a Jupyter environment and run its cells.

## Analysis workflow

### 1. Cohort construction

The pipeline selects records for female patients using `gender_concept_id = 8532` and creates demographic cohorts using race and ethnicity concept IDs:

- Black females
- Hispanic females
- Asian females
- Non-Hispanic White females

Conditions observed in fewer than 10 distinct patients within a cohort are excluded by default. Cohort definitions and the minimum patient count are configurable in `src/cohort.py` and in the main script.

### 2. Condition network construction

For each cohort, the workflow builds an undirected graph:

- **Condition nodes** represent OMOP condition concept IDs.
- **Condition co-occurrence edges** connect conditions recorded during the same patient visit. Duplicate occurrences of a condition within a visit are counted once.
- **Edge attributes** include the raw co-occurrence count and pointwise mutual information (PMI).
- **Mortality node** (`NODE_DEATH`) is added when the cohort contains patients present in `death.csv`. It is connected to conditions recorded for those patients.

The case-fatality co-occurrence ratio for a condition is calculated as the number of deceased patients with that condition divided by the number of patients with that condition in the cohort. This is an observational association, not a causal or predictive estimate.

### 3. Network metrics

The analysis calculates condition-node metrics for each cohort:

- Degree and degree centrality
- Weighted degree, using PMI-derived edge weights
- Betweenness centrality
- Louvain community assignment and graph modularity
- Case-fatality ratio and fatality count

The centrality and community metrics are calculated on the condition-only subgraph; the mortality node is excluded from those calculations.

### 4. Demographic comparison

The workflow combines cohort graphs into a comparison graph and calculates pairwise Jaccard overlap among the top 20 diagnoses, ranked by betweenness centrality.

A permutation-based modularity test is implemented in `src/network_metrics.py`, but it is not run by default in `ehrshot_network_analysis.py`.

### 5. Visualization

The pipeline renders cohort-specific and demographic-comparison network figures using a force-directed (Fruchterman–Reingold) layout.

In cohort figures, node size reflects betweenness centrality, node color represents Louvain community assignment, and mortality edges are distinguished from condition co-occurrence edges. The comparison figure uses edge colors to distinguish demographic cohorts.

## Outputs

The main script writes the following files to `outputs/`:

- `nodes.csv` and `nodes_<cohort>.csv` — condition and mortality node identifiers
- `edges_all_cohorts.csv` — combined edge list with a cohort column
- `edges_<cohort>.csv` — edge list for each cohort
- `<cohort>_network.png` — network visualization for each cohort
- `demographic_comparison.graphml` — cohort comparison graph, including cohort-specific edge attributes
- `demographic_comparison_network.png` — visualization of the comparison graph
- `top_diagnosis_jaccard_overlap.csv` — pairwise overlap among top diagnoses
- `node_mapping.csv` — mapping of node IDs to human-readable OMOP concept metadata

Outputs are regenerated from the input data when the analysis script is run.

## Reproducibility and interpretation

The graph layout and Louvain community detection use fixed random seeds to make results more reproducible. Results still depend on the input data, cohort definitions, and analysis parameters.

Condition nodes are labeled with concept IDs in the current exports and visualizations. Mapping those IDs to human-readable concept names may be added as a separate step if a suitable OMOP concept table is available.
