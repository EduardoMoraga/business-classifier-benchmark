from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import classification_metrics
from src.features.preprocess import build_preprocessor
from src.features.selection import numeric_correlation_report, vif_report
from src.models.typed_state import employee_row_to_state

TARGET = "Attrition"
RANDOM_STATE = 42
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
SEGMENT_COLUMNS = ["OverTime", "BusinessTravel", "MaritalStatus", "Department", "JobRole"]
LOGISTIC_C = 0.25


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
                    C=LOGISTIC_C,
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


def person_drivers(model: Pipeline, x: pd.DataFrame, top: int = 3) -> list[dict]:
    """Per-employee logit contributions (coefficient x transformed value), named by raw value."""
    preprocessor = model.named_steps["preprocess"]
    contrib = preprocessor.transform(x) * model.named_steps["model"].coef_[0]
    names = preprocessor.get_feature_names_out()
    columns = sorted(x.columns, key=len, reverse=True)

    def label(feature: str, row: pd.Series) -> str:
        raw = feature.split("__", 1)[1]
        column = raw if raw in x.columns else next(c for c in columns if raw.startswith(f"{c}_"))
        return f"{column} = {row[column]}"

    drivers = []
    for i, (_, row) in enumerate(x.iterrows()):
        order = np.argsort(contrib[i])
        drivers.append({
            "raises": [label(names[j], row) for j in order[::-1][:top] if contrib[i, j] > 0],
            "lowers": [label(names[j], row) for j in order[:top] if contrib[i, j] < 0],
        })
    return drivers


def hybrid_state(state: str, score: float, percentile: float, drivers: dict) -> str:
    """Typed state plus the validated model's view, for the LLM-as-deployment-layer test."""
    raises = "; ".join(drivers["raises"]) or "none"
    lowers = "; ".join(drivers["lowers"]) or "none"
    return (
        f"{state} Statistical model context: a validated logistic regression trained on this "
        f"company's history gives this employee an attrition score of {score:.2f}, higher than "
        f"{percentile:.0f}% of training employees. Scores are well ordered but inflated, so read "
        f"them as a ranking. Factors raising risk: {raises}. Factors lowering risk: {lowers}."
    )


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


def segment_rates(x: pd.DataFrame, y: pd.Series) -> list[dict]:
    """Attrition rate by business segment, computed on training rows only."""
    rows = []
    for column in SEGMENT_COLUMNS:
        if column not in x.columns:
            continue
        grouped = y.groupby(x[column]).agg(["mean", "size"])
        for level, stats in grouped.iterrows():
            rows.append(
                {
                    "segment": column,
                    "level": str(level),
                    "rate": float(stats["mean"]),
                    "n": int(stats["size"]),
                }
            )
    return rows


def _downsample(points: list[tuple[float, float]], max_points: int = 80) -> list[list[float]]:
    if len(points) > max_points:
        idx = np.unique(np.linspace(0, len(points) - 1, max_points).round().astype(int))
        points = [points[i] for i in idx]
    return [[round(float(a), 4), round(float(b), 4)] for a, b in points]


def curve_summary(y_true, y_score) -> dict:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "pr": _downsample(list(zip(recall[::-1], precision[::-1]))),
        "roc": _downsample(list(zip(fpr, tpr))),
    }


def calibration_summary(y_true, y_score, bins: int = 8) -> list[dict]:
    """Reliability table with quantile bins: mean predicted vs observed rate."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    frame["bin"] = pd.qcut(frame["score"], q=bins, duplicates="drop")
    table = frame.groupby("bin", observed=True).agg(
        predicted=("score", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return [
        {"predicted": float(r.predicted), "observed": float(r.observed), "n": int(r.n)}
        for r in table.itertuples()
    ]


def threshold_sweep(y_true, y_score) -> list[dict]:
    """Confusion counts for each threshold between 0.05 and 0.95."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = y_score >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "tp": int((pred & y_true).sum()),
                "fp": int((pred & ~y_true).sum()),
                "fn": int((~pred & y_true).sum()),
                "tn": int((~pred & ~y_true).sum()),
            }
        )
    return rows


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

    holdout_rows = test_x.copy()
    holdout_rows.insert(0, "row_id", test_x.index.astype(int))
    holdout_rows["y_true"] = test_y.map({1: True, 0: False}).to_numpy()
    holdout_rows["classical_score"] = logistic_score
    holdout_rows["typed_state"] = [employee_row_to_state(row) for _, row in test_x.iterrows()]
    train_scores = np.sort(logistic.predict_proba(train_x)[:, 1])
    percentiles = 100 * np.searchsorted(train_scores, logistic_score) / len(train_scores)
    drivers = person_drivers(logistic, test_x)
    holdout_rows["model_percentile"] = percentiles
    holdout_rows["model_drivers"] = drivers
    holdout_rows["hybrid_state"] = [
        hybrid_state(state, score, pct, drv)
        for state, score, pct, drv in zip(
            holdout_rows["typed_state"], logistic_score, percentiles, drivers, strict=True
        )
    ]
    sample_rows = holdout_rows.head(12)

    dataset_fingerprint = hashlib.sha256(raw_df.to_csv(index=False).encode()).hexdigest()[:16]

    summary = {
        "run": {
            "id": f"attrition-lab-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_hash": dataset_fingerprint,
        },
        "dataset": "IBM HR Analytics Employee Attrition",
        "source": "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "rows": len(raw_df),
        "columns": int(raw_df.shape[1]),
        "target": TARGET,
        "split": {"train": len(train_x), "test": len(test_x), "random_state": RANDOM_STATE},
        "eda": {**eda_summary(train_x, train_y), "scope": "train_only"},
        "segments": segment_rates(train_x, train_y),
        "pca": pca_summary(train_x),
        "cv": cv,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(test_y, y_pred).tolist(),
        "curves": curve_summary(test_y, logistic_score),
        "calibration": calibration_summary(test_y, logistic_score),
        "threshold_sweep": threshold_sweep(test_y, logistic_score),
        "model_config": {
            "estimator": "LogisticRegression",
            "solver": "liblinear",
            "penalty": "l1",
            "C": LOGISTIC_C,
            "class_weight": "balanced",
            "preprocessing": "OneHotEncoder (categorical) + StandardScaler (numeric)",
            "dropped_columns": DROP_COLUMNS,
            "cv": "RepeatedStratifiedKFold 5x5",
            "test_size": 0.2,
        },
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
    (paths["app_public"] / "holdout.json").write_text(
        holdout_rows.to_json(orient="records", indent=1)
    )
    (paths["app_public"] / "holdout-sample.json").write_text(
        sample_rows.to_json(orient="records", indent=2)
    )
    pd.DataFrame(metrics).to_csv(paths["reports"] / "model_metrics.csv", index=False)
    coef.to_csv(paths["reports"] / "logistic_coefficients.csv", index=False)
    importances.to_csv(paths["reports"] / "permutation_importance.csv", index=False)
    joblib.dump(logistic, paths["models"] / "logistic_attrition.joblib")

    return summary
