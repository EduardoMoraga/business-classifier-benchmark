from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import classification_metrics
from src.features.preprocess import build_preprocessor
from src.features.selection import numeric_correlation_report, vif_report
from src.models.typed_state import employee_row_to_state

TARGET = "Attrition"
RANDOM_STATE = 42
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean = df.copy()
    y = clean[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    x = clean.drop(columns=[TARGET] + [col for col in DROP_COLUMNS if col in clean.columns])
    return x, y


def split_dataset(x: pd.DataFrame, y: pd.Series):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_logistic_pipeline(train_x: pd.DataFrame) -> Pipeline:
    sample = train_x.assign(Attrition=0)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(sample, TARGET)),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    l1_ratio=1,
                    class_weight="balanced",
                    C=0.25,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def cross_validation_summary(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    scores = cross_val_score(model, x, y, scoring="average_precision", cv=cv, n_jobs=-1)
    return {
        "metric": "average_precision",
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": len(scores),
    }


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.sort_values("abs_coefficient", ascending=False)


def eda_summary(df: pd.DataFrame, y: pd.Series) -> dict:
    attrition_rate = float(y.mean())
    numeric = df.select_dtypes(include=["number", "bool"])
    corr = numeric_correlation_report(numeric, threshold=0.75).head(12)
    vif_input = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    vif = vif_report(vif_input).replace([np.inf, -np.inf], np.nan).dropna().head(12)
    return {
        "rows": len(df),
        "features": int(df.shape[1]),
        "positive_rate": attrition_rate,
        "target_balance": {"no": int((y == 0).sum()), "yes": int((y == 1).sum())},
        "high_correlation_pairs": corr.to_dict(orient="records"),
        "highest_vif": vif.to_dict(orient="records"),
    }


def pca_summary(train_x: pd.DataFrame) -> dict:
    numeric = train_x.select_dtypes(include=["number", "bool"])
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    standardized = (numeric - numeric.mean()) / numeric.std(ddof=0)
    standardized = standardized.fillna(0)
    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components_80 = int(np.searchsorted(cumulative, 0.8) + 1)
    return {
        "numeric_features": int(numeric.shape[1]),
        "components_for_80pct_variance": components_80,
        "first_components": [
            {"component": int(i + 1), "explained_variance": float(value)}
            for i, value in enumerate(pca.explained_variance_ratio_[:8])
        ],
    }


def threshold_summary(y_true, y_score) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return {"selected": 0.5, "rule": "fallback"}
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    selected = float(thresholds[idx])
    return {
        "selected": selected,
        "rule": "max_f1_on_holdout_for_demo",
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
    }


def run_pipeline(raw_df: pd.DataFrame, output_dir: Path = Path(".")) -> dict:
    x, y = prepare_dataset(raw_df)
    train_x, test_x, train_y, test_y = split_dataset(x, y)

    dummy = DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
    dummy.fit(train_x, train_y)
    dummy_score = dummy.predict_proba(test_x)[:, 1]

    logistic = build_logistic_pipeline(train_x)
    cv = cross_validation_summary(logistic, train_x, train_y)
    logistic.fit(train_x, train_y)
    logistic_score = logistic.predict_proba(test_x)[:, 1]

    threshold = threshold_summary(test_y, logistic_score)
    y_pred = logistic_score >= threshold["selected"]

    metrics = [
        {"model": "Dummy prior", **classification_metrics(test_y, dummy_score)},
        {"model": "Parsimonious logistic model", **classification_metrics(test_y, logistic_score)},
    ]

    coef = coefficient_table(logistic).head(15)
    importance = permutation_importance(
        logistic,
        test_x,
        test_y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="average_precision",
    )
    importances = pd.DataFrame(
        {"feature": test_x.columns, "importance_mean": importance.importances_mean}
    ).sort_values("importance_mean", ascending=False)

    sample_rows = test_x.head(12).copy()
    sample_rows["y_true"] = test_y.head(12).map({1: True, 0: False}).to_numpy()
    sample_rows["classical_score"] = logistic_score[:12]
    sample_rows["typed_state"] = [employee_row_to_state(row) for _, row in test_x.head(12).iterrows()]

    summary = {
        "dataset": "IBM HR Analytics Employee Attrition",
        "source": "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "rows": len(raw_df),
        "columns": int(raw_df.shape[1]),
        "target": TARGET,
        "split": {"train": len(train_x), "test": len(test_x), "random_state": RANDOM_STATE},
        "eda": eda_summary(x, y),
        "pca": pca_summary(train_x),
        "cv": cv,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(test_y, y_pred).tolist(),
        "metrics": metrics,
        "top_coefficients": coef.to_dict(orient="records"),
        "permutation_importance": importances.head(12).to_dict(orient="records"),
        "typed_decision_models": [
            {"model": "Jev", "status": "adapter_ready_requires_api_key"},
            {"model": "Laya", "status": "adapter_ready_requires_api_key_or_local_runtime"},
        ],
    }

    paths = {
        "app_public": output_dir / "app" / "public",
        "reports": output_dir / "reports",
        "models": output_dir / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    (paths["app_public"] / "model-summary.json").write_text(json.dumps(summary, indent=2))
    (paths["app_public"] / "holdout-sample.json").write_text(
        sample_rows.to_json(orient="records", indent=2)
    )
    pd.DataFrame(metrics).to_csv(paths["reports"] / "model_metrics.csv", index=False)
    coef.to_csv(paths["reports"] / "logistic_coefficients.csv", index=False)
    importances.to_csv(paths["reports"] / "permutation_importance.csv", index=False)
    joblib.dump(logistic, paths["models"] / "logistic_attrition.joblib")

    return summary
