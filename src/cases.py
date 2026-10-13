"""The two attrition datasets the benchmark runs on, and how each one is modelled."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.load_dataset import HR_RAW_PATH, RAW_PATH, load_attrition_csv, load_hr_workbook
from src.features.engineering import hr_features


@dataclass(frozen=True)
class Case:
    id: str
    dataset: str
    source: str
    raw_path: Path
    load: Callable[[Path], pd.DataFrame]
    production: str  # "logistic" (parsimonious L1) or "xgboost" (the original notebook's model)
    model_label: str  # how the hybrid state names the production model to the LLM
    score_note: str  # what the LLM is told about the score scale (checked against calibration)
    drop_columns: list[str] = field(default_factory=list)
    segment_columns: list[str] = field(default_factory=list)
    dedupe: bool = False
    features: Callable[[pd.DataFrame], pd.DataFrame] | None = None


CASES = {
    "ibm": Case(
        id="ibm",
        dataset="IBM HR Analytics Employee Attrition",
        source="https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        raw_path=RAW_PATH,
        load=load_attrition_csv,
        production="logistic",
        model_label="logistic regression",
        score_note="Scores are well ordered but inflated, so read them as a ranking.",
        drop_columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
        segment_columns=["OverTime", "BusinessTravel", "MaritalStatus", "Department", "JobRole"],
    ),
    "hr": Case(
        id="hr",
        dataset="HR Analytics (employee-attrition-predictor)",
        source="https://github.com/EduardoMoraga/employee-attrition-predictor",
        raw_path=HR_RAW_PATH,
        load=load_hr_workbook,
        production="xgboost",
        model_label="gradient-boosted tree model (XGBoost)",
        score_note="Scores are close to calibrated probabilities on held-out employees.",
        segment_columns=["number_project", "salary", "department", "time_spend_company",
                         "promotion_last_5years"],
        # 3,008 rows are exact copies; left in, they would sit on both sides of the split.
        dedupe=True,
        features=hr_features,
    ),
}


def load_case(case: Case, project_root: Path) -> tuple[pd.DataFrame, int]:
    """Raw rows for a case and how many exact duplicates were dropped."""
    raw = case.load(project_root / case.raw_path)
    if not case.dedupe:
        return raw, 0
    clean = raw.drop_duplicates()
    return clean, len(raw) - len(clean)
