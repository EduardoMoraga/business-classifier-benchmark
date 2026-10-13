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
from xgboost import XGBClassifier

from src.cases import CASES, Case
from src.evaluation.metrics import classification_metrics
from src.features.selection import numeric_correlation_report, vif_report
from src.models.pipelines import TARGET, build_model_pipeline, contributions, engineered_frame
from src.models.typed_state import employee_row_to_state

RANDOM_STATE = 42
DROP_COLUMNS = CASES["ibm"].drop_columns
LOGISTIC_C = 0.25
# Fixed XGBoost configuration from a prior analysis of the HR dataset (employee-attrition-predictor),
# set before seeing the holdout. That analysis early-stopped on its test set; here the tree count is fixed.
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def prepare_dataset(df: pd.DataFrame, drop_columns: list[str] = DROP_COLUMNS):
    clean = df.copy()
    y = clean[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    x = clean.drop(columns=[TARGET] + [col for col in drop_columns if col in clean.columns])
    return x, y


def split_dataset(x: pd.DataFrame, y: pd.Series):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_logistic_pipeline(train_x: pd.DataFrame, features=None) -> Pipeline:
    model = LogisticRegression(
        solver="liblinear",
        l1_ratio=1,
        class_weight="balanced",
        C=LOGISTIC_C,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return build_model_pipeline(train_x, model, features)


def build_xgboost_pipeline(train_x: pd.DataFrame, train_y: pd.Series, features=None) -> Pipeline:
    model = XGBClassifier(
        **XGBOOST_PARAMS,
        scale_pos_weight=float((train_y == 0).sum() / (train_y == 1).sum()),
        eval_metric="logloss",
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    return build_model_pipeline(train_x, model, features)


def build_production_model(case: Case, train_x: pd.DataFrame, train_y: pd.Series) -> Pipeline:
    if case.production == "xgboost":
        return build_xgboost_pipeline(train_x, train_y, case.features)
    return build_logistic_pipeline(train_x, case.features)


def model_config(case: Case) -> dict:
    common = {
        "preprocessing": "OneHotEncoder (categorical) + StandardScaler (numeric)",
        "dropped_columns": case.drop_columns,
        "engineered_features": ["satisfaction_x_evaluation", "hours_per_project", "overwork"]
        if case.features else [],
        "deduplicated": case.dedupe,
        "cv": "RepeatedStratifiedKFold 5x5",
        "test_size": 0.2,
    }
    if case.production == "xgboost":
        return {"estimator": "XGBClassifier", **XGBOOST_PARAMS,
                "class_weight": "scale_pos_weight = negatives / positives", **common}
    return {"estimator": "LogisticRegression", "solver": "liblinear", "penalty": "l1",
            "C": LOGISTIC_C, "class_weight": "balanced", **common}


def cross_validation_summary(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    scores = cross_val_score(model, x, y, scoring="average_precision", cv=cv, n_jobs=-1)
    return {
        "metric": "average_precision",
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": len(scores),
    }


def coefficient_table(model: Pipeline) -> pd.DataFrame | None:
    if not isinstance(model.named_steps["model"], LogisticRegression):
        return None
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.sort_values("abs_coefficient", ascending=False)


def _show(value) -> str:
    if isinstance(value, float | np.floating) and not float(value).is_integer():
        return f"{value:.2f}"
    return str(value.item() if isinstance(value, np.generic) else value)


def person_drivers(model: Pipeline, x: pd.DataFrame, top: int = 3) -> list[dict]:
    """Per-employee log-odds contributions, named by the employee's raw value."""
    contrib, names = contributions(model, x)
    frame = engineered_frame(model, x)
    columns = sorted(frame.columns, key=len, reverse=True)

    def label(feature: str, row: pd.Series) -> str:
        raw = feature.split("__", 1)[1]
        column = raw if raw in frame.columns else next(c for c in columns if raw.startswith(f"{c}_"))
        return f"{column} = {_show(row[column])}"

    drivers = []
    for i, (_, row) in enumerate(frame.iterrows()):
        order = np.argsort(contrib[i])
        drivers.append({
            "raises": [label(names[j], row) for j in order[::-1][:top] if contrib[i, j] > 0],
            "lowers": [label(names[j], row) for j in order[:top] if contrib[i, j] < 0],
        })
    return drivers


def global_drivers(model: Pipeline, x: pd.DataFrame, top: int = 12) -> list[dict]:
    """Mean |contribution| per feature and whether high values push risk up (train rows)."""
    contrib, names = contributions(model, x)
    transformed = model.named_steps["preprocess"].transform(engineered_frame(model, x))
    rows = []
    for j, name in enumerate(names):
        spread = np.std(transformed[:, j]) > 0 and np.std(contrib[:, j]) > 0
        direction = float(np.corrcoef(transformed[:, j], contrib[:, j])[0, 1]) if spread else 0.0
        rows.append({"feature": name, "mean_abs": float(np.abs(contrib[:, j]).mean()),
                     "direction": direction})
    return sorted(rows, key=lambda r: -r["mean_abs"])[:top]


def hybrid_state(
    state: str,
    score: float,
    percentile: float,
    drivers: dict,
    model_label: str = "logistic regression",
    score_note: str = "Scores are well ordered but inflated, so read them as a ranking.",
) -> str:
    """Typed state plus the validated model's view, for the LLM-as-deployment-layer test."""
    raises = "; ".join(drivers["raises"]) or "none"
    lowers = "; ".join(drivers["lowers"]) or "none"
    return (
        f"{state} Statistical model context: a validated {model_label} trained on this "
        f"company's history gives this employee an attrition score of {score:.2f}, higher than "
        f"{percentile:.0f}% of training employees. {score_note} "
        f"Factors raising risk: {raises}. Factors lowering risk: {lowers}."
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


def segment_rates(x: pd.DataFrame, y: pd.Series, columns: list[str]) -> list[dict]:
    """Attrition rate by business segment, computed on training rows only."""
    rows = []
    for column in columns:
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


def run_pipeline(
    raw_df: pd.DataFrame,
    output_dir: Path = Path("."),
    case: Case = CASES["ibm"],
    duplicates_removed: int = 0,
) -> dict:
    x, y = prepare_dataset(raw_df, case.drop_columns)
    train_x, test_x, train_y, test_y = split_dataset(x, y)

    dummy = DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
    dummy.fit(train_x, train_y)
    dummy_score = dummy.predict_proba(test_x)[:, 1]

    production = build_production_model(case, train_x, train_y)
    cv = cross_validation_summary(production, train_x, train_y)
    production.fit(train_x, train_y)
    production_score = production.predict_proba(test_x)[:, 1]

    production_name = (
        "XGBoost (baseline configuration)" if case.production == "xgboost"
        else "Parsimonious logistic model"
    )
    threshold = threshold_summary(test_y, production_score)
    y_pred = production_score >= threshold["selected"]

    metrics = [
        {"model": "Dummy prior", **classification_metrics(test_y, dummy_score)},
        {"model": production_name, **classification_metrics(test_y, production_score)},
    ]

    coef = coefficient_table(production)
    coef = coef.head(15) if coef is not None else None
    importance = permutation_importance(
        production,
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
    holdout_rows["classical_score"] = production_score
    holdout_rows["typed_state"] = [employee_row_to_state(row) for _, row in test_x.iterrows()]
    train_scores = np.sort(production.predict_proba(train_x)[:, 1])
    percentiles = 100 * np.searchsorted(train_scores, production_score) / len(train_scores)
    drivers = person_drivers(production, test_x)
    holdout_rows["model_percentile"] = percentiles
    holdout_rows["model_drivers"] = drivers
    holdout_rows["hybrid_state"] = [
        hybrid_state(state, score, pct, drv, case.model_label, case.score_note)
        for state, score, pct, drv in zip(
            holdout_rows["typed_state"], production_score, percentiles, drivers, strict=True
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
        "case": case.id,
        "dataset": case.dataset,
        "source": case.source,
        "production": case.production,
        "production_name": production_name,
        "duplicates_removed": duplicates_removed,
        "rows": len(raw_df),
        "columns": int(raw_df.shape[1]),
        "target": TARGET,
        "split": {"train": len(train_x), "test": len(test_x), "random_state": RANDOM_STATE},
        "eda": {**eda_summary(train_x, train_y), "scope": "train_only"},
        "segments": segment_rates(train_x, train_y, case.segment_columns),
        "pca": pca_summary(train_x),
        "cv": cv,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(test_y, y_pred).tolist(),
        "curves": curve_summary(test_y, production_score),
        "calibration": calibration_summary(test_y, production_score),
        "threshold_sweep": threshold_sweep(test_y, production_score),
        "model_config": model_config(case),
        "metrics": metrics,
        "top_coefficients": coef.to_dict(orient="records") if coef is not None else [],
        "top_drivers": global_drivers(production, train_x),
        "permutation_importance": importances.head(12).to_dict(orient="records"),
        "typed_decision_models": [
            {"model": "Jev", "status": "adapter_ready_requires_api_key"},
            {"model": "Laya", "status": "adapter_ready_requires_api_key_or_local_runtime"},
        ],
    }

    paths = {
        "app_public": output_dir / "app" / "public" / case.id,
        "reports": output_dir / "reports" / case.id,
        "models": output_dir / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    (paths["app_public"] / "model-summary.json").write_text(json.dumps(summary, indent=2))
    (paths["app_public"] / "holdout.json").write_text(
        holdout_rows.to_json(orient="records")
    )
    (paths["app_public"] / "holdout-sample.json").write_text(
        sample_rows.to_json(orient="records", indent=2)
    )
    pd.DataFrame(metrics).to_csv(paths["reports"] / "model_metrics.csv", index=False)
    if coef is not None:
        coef.to_csv(paths["reports"] / "logistic_coefficients.csv", index=False)
    importances.to_csv(paths["reports"] / "permutation_importance.csv", index=False)
    joblib.dump(production, paths["models"] / f"{case.id}_{case.production}.joblib")

    return summary
