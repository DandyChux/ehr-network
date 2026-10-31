from __future__ import annotations

from pathlib import Path

import pandas as pd

FEMALE_CONCEPT_ID = 8532

DEFAULT_COHORT_DEFINITIONS = {
    "black_female": {
        "race_concept_ids": {8516},
        "ethnicity_concept_ids": None,
        "exclude_ethnicity_concept_ids": {38003563},
    },
    "hispanic_female": {
        "race_concept_ids": None,
        "ethnicity_concept_ids": {38003563},
    },
    "asian_female": {
        "race_concept_ids": {8515},
        "ethnicity_concept_ids": None,
        "exclude_ethnicity_concept_ids": {38003563},
    },
    "non_hispanic_white_female": {
        "race_concept_ids": {8527},
        "ethnicity_concept_ids": None,
        "exclude_ethnicity_concept_ids": {38003563},
    },
}


def _read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing OMOP table: {path}")

    return pd.read_csv(path)


def load_omop_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)

    return {
        "person": _read_csv(data_dir, "person.csv"),
        "condition_occurrence": _read_csv(data_dir, "condition_occurrence.csv"),
        "visit_occurrence": _read_csv(data_dir, "visit_occurrence.csv"),
        "death": _read_csv(data_dir, "death.csv"),
    }


def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")


def build_cohorts(
    tables: dict[str, pd.DataFrame],
    cohort_definitions: dict | None = None,
    min_patient_count: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Build female demographic cohorts.

    Returned rows represent condition occurrences and include:
    - person_id
    - visit_occurrence_id
    - condition_concept_id
    - deceased
    - cohort
    """

    cohort_definitions = cohort_definitions or DEFAULT_COHORT_DEFINITIONS

    person = tables["person"].copy()
    conditions = tables["condition_occurrence"].copy()
    visits = tables["visit_occurrence"].copy()
    death = tables["death"].copy()

    _require_columns(
        person,
        {
            "person_id",
            "gender_concept_id",
            "race_concept_id",
            "ethnicity_concept_id",
        },
        "person.csv",
    )

    _require_columns(
        conditions,
        {
            "person_id",
            "visit_occurrence_id",
            "condition_concept_id",
        },
        "condition_occurrence.csv",
    )

    _require_columns(
        visits,
        {"person_id", "visit_occurrence_id"},
        "visit_occurrence.csv",
    )

    _require_columns(death, {"person_id"}, "death.csv")

    female = person.loc[person["gender_concept_id"].eq(FEMALE_CONCEPT_ID)].copy()

    deceased_ids = set(death["person_id"].dropna().unique())

    female["deceased"] = female["person_id"].isin(deceased_ids)

    # Restrict condition rows to patients with valid female demographics.
    events = conditions.merge(
        female[
            [
                "person_id",
                "race_concept_id",
                "ethnicity_concept_id",
                "deceased",
            ]
        ],
        on="person_id",
        how="inner",
    )

    # Validate that condition visits belong to the same patient.
    events = events.merge(
        visits[["visit_occurrence_id", "person_id"]].drop_duplicates(),
        on=["visit_occurrence_id", "person_id"],
        how="inner",
    )

    output: dict[str, pd.DataFrame] = {}

    for cohort_name, definition in cohort_definitions.items():
        mask = pd.Series(True, index=events.index)

        race_ids = definition.get("race_concept_ids")
        if race_ids is not None:
            mask &= events["race_concept_id"].isin(race_ids)

        ethnicity_ids = definition.get("ethnicity_concept_ids")
        if ethnicity_ids is not None:
            mask &= events["ethnicity_concept_id"].isin(ethnicity_ids)

        excluded_ethnicity_ids = definition.get("exclude_ethnicity_concept_ids")
        if excluded_ethnicity_ids is not None:
            mask &= ~events["ethnicity_concept_id"].isin(excluded_ethnicity_ids)

        cohort_events = events.loc[mask].copy()

        if cohort_events.empty:
            continue

        # Remove conditions occurring in fewer than N patients.
        patient_counts = cohort_events.groupby("condition_concept_id")[
            "person_id"
        ].nunique()

        retained_concepts = patient_counts[patient_counts >= min_patient_count].index

        cohort_events = cohort_events[
            cohort_events["condition_concept_id"].isin(retained_concepts)
        ].copy()

        if cohort_events.empty:
            continue

        cohort_events["cohort"] = cohort_name
        output[cohort_name] = cohort_events.reset_index(drop=True)

    return output
