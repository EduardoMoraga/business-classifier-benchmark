"""Model tournament: can a more complex classifier beat the parsimonious logistic model?

Every candidate is tuned on the training split only (inner 5-fold randomized search), scored
with the same RepeatedStratifiedKFold 5x5 as the production model, and evaluated once on the
untouched holdout. Feature selection (Boruta) runs inside the pipeline, so it never sees the
fold it is evaluated on.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.boruta_selector import BorutaSelector
from src.models.pipelines import build_model_pipeline

RANDOM_STATE = 42
SEARCH_ITER = 25
BOOTSTRAP_ROUNDS = 1000
PRODUCTION_ID = "logistic_l1"
ORIGINAL_ID = "xgboost_original"
BORUTA_CACHE = Path(".cache/boruta")


@dataclass
class Candidate:
    id: str
    name: str
    family: str
    plain: str
    estimator: object
    grid: dict = field(default_factory=dict)


def build_candidates(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    features=None,
    original_xgboost: dict | None = None,
    reference_label: str = "actual",
) -> list[Candidate]:
    """original_xgboost: fixed hyperparameters of a pre-existing XGBoost to score untuned."""
    pos_weight = float((train_y == 0).sum() / (train_y == 1).sum())

    def _pipe(model, boruta: bool = False) -> Pipeline:
        # Boruta only depends on the fold, so cache it across the hyperparameter search.
        return build_model_pipeline(
            train_x, model, features,
            selector=BorutaSelector(random_state=RANDOM_STATE) if boruta else None,
            memory=str(BORUTA_CACHE) if boruta else None,
        )

    def xgb(**params):
        return XGBClassifier(
            **({"n_estimators": 400} | params),
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    def logistic(**kwargs):
        return LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, **kwargs)

    boost_grid = {
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1, 5, 10],
    }
    candidates = [
        Candidate(
            "dummy", "Sin modelo (dummy)", "Referencia",
            "Asigna a todos la tasa histórica de salida.",
            DummyClassifier(strategy="prior"),
        ),
        Candidate(
            PRODUCTION_ID, f"Regresión logística L1 ({reference_label})", "Lineal",
            "Suma ponderada de factores; elimina los que no aportan.",
            _pipe(logistic(solver="liblinear", l1_ratio=1, C=0.25,
                                             class_weight="balanced")),
        ),
        Candidate(
            "logistic_unbalanced", "Regresión logística sin balancear", "Lineal",
            "La misma idea, pero optimizada para acertar en general y no para detectar salidas.",
            _pipe(logistic(solver="liblinear", l1_ratio=1, C=0.25)),
        ),
        Candidate(
            "logistic_l2", "Regresión logística L2 ajustada", "Lineal",
            "Logística con la fuerza de regularización elegida por validación.",
            _pipe(logistic(class_weight="balanced")),
            {"model__C": np.logspace(-3, 1, 20)},
        ),
        Candidate(
            "random_forest", "Random Forest", "Árboles",
            "Cientos de árboles de decisión que votan.",
            _pipe(RandomForestClassifier(
                n_estimators=500, class_weight="balanced_subsample", n_jobs=1,
                random_state=RANDOM_STATE)),
            {
                "model__max_depth": [4, 6, 8, 12, None],
                "model__min_samples_leaf": [1, 3, 5, 10, 20],
                "model__max_features": ["sqrt", 0.3, 0.5],
            },
        ),
        Candidate(
            "xgboost", "XGBoost", "Boosting",
            "Árboles en secuencia: cada uno corrige los errores del anterior.",
            _pipe(xgb()),
            boost_grid,
        ),
        Candidate(
            "lightgbm", "LightGBM", "Boosting",
            "Boosting rápido de árboles, muy usado en competencias.",
            _pipe(LGBMClassifier(
                n_estimators=400, class_weight="balanced", subsample_freq=1, n_jobs=1,
                random_state=RANDOM_STATE, verbose=-1)),
            {
                "model__num_leaves": [4, 8, 15, 31],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__min_child_samples": [10, 20, 40],
                "model__subsample": [0.7, 1.0],
                "model__colsample_bytree": [0.5, 0.7, 1.0],
                "model__reg_lambda": [0, 5, 10],
            },
        ),
        Candidate(
            "boruta_logistic", "Boruta + logística", "Selección de variables",
            "Boruta deja sólo las variables que superan a copias aleatorias de sí mismas.",
            _pipe(logistic(class_weight="balanced"), boruta=True),
            {"model__C": np.logspace(-3, 1, 12)},
        ),
        Candidate(
            "boruta_xgboost", "Boruta + XGBoost", "Selección de variables",
            "XGBoost entrenado sólo con las variables que confirma Boruta.",
            _pipe(xgb(), boruta=True),
            {
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.01, 0.03, 0.05],
                "model__subsample": [0.7, 1.0],
                "model__colsample_bytree": [0.7, 1.0],
            },
        ),
    ]
    if original_xgboost:
        candidates.insert(2, Candidate(
            ORIGINAL_ID, "XGBoost del ejercicio original", "Boosting",
            "Tu XGBoost con sus hiperparámetros originales, sin ajustar y sin fugas.",
            _pipe(xgb(**original_xgboost)),
        ))
    return candidates


def _tune(candidate: Candidate, train_x, train_y):
    if not candidate.grid:
        return clone(candidate.estimator).fit(train_x, train_y), {}
    space = int(np.prod([len(v) for v in candidate.grid.values()]))
    search = RandomizedSearchCV(
        candidate.estimator,
        candidate.grid,
        n_iter=min(SEARCH_ITER, space),
        scoring="average_precision",
        cv=StratifiedKFold(5, shuffle=True, random_state=7),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(train_x, train_y)
    params = {k.removeprefix("model__"): _plain(v) for k, v in search.best_params_.items()}
    return search.best_estimator_, params


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _n_features(model) -> int | None:
    if not isinstance(model, Pipeline):
        return None
    if "boruta" in model.named_steps:
        return int(model.named_steps["boruta"].support_.sum())
    return len(model.named_steps["preprocess"].get_feature_names_out())


def holdout_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_score >= threshold, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_score >= threshold)),
        "confusion": {"threshold": threshold, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def paired_bootstrap(y_true, score_a, score_b, rounds: int = BOOTSTRAP_ROUNDS, seed: int = RANDOM_STATE):
    """95% interval of PR-AUC(a) - PR-AUC(b) over holdout resamples that keep both classes."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    diffs = []
    while len(diffs) < rounds:
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(
            average_precision_score(y_true[idx], score_a[idx])
            - average_precision_score(y_true[idx], score_b[idx])
        )
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "share_a_wins": float(np.mean(np.array(diffs) > 0))}


def run_tournament(
    train_x,
    train_y,
    holdout_x,
    holdout_y,
    features=None,
    original_xgboost: dict | None = None,
    deployed: str = PRODUCTION_ID,
    reference_label: str = "actual",
    log=print,
) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    y_true = holdout_y.to_numpy()
    fitted, rows = {}, []

    for candidate in build_candidates(train_x, train_y, features, original_xgboost, reference_label):
        started = time.perf_counter()
        model, params = _tune(candidate, train_x, train_y)
        cv = cross_val_score(clone(model), train_x, train_y, scoring="average_precision",
                             cv=outer, n_jobs=-1)
        score = model.predict_proba(holdout_x)[:, 1]
        fitted[candidate.id] = (model, score)
        rows.append({
            "id": candidate.id,
            "name": candidate.name,
            "family": candidate.family,
            "plain": candidate.plain,
            "params": params,
            "n_features": _n_features(model),
            "cv": {"mean": float(cv.mean()), "std": float(cv.std()), "folds": len(cv)},
            "holdout": holdout_metrics(y_true, score),
            "seconds": round(time.perf_counter() - started, 1),
        })
        log(f"{candidate.name}: CV PR-AUC {cv.mean():.3f} ± {cv.std():.3f}, "
            f"holdout {rows[-1]['holdout']['pr_auc']:.3f}")

    # Soft-vote the production logistic with the best tree model: the usual "best of both" try.
    trees = [r for r in rows if r["family"] in ("Árboles", "Boosting")]
    best_tree = max(trees, key=lambda r: r["cv"]["mean"])
    started = time.perf_counter()
    ensemble = VotingClassifier(
        [("logistic", fitted[PRODUCTION_ID][0]), ("tree", fitted[best_tree["id"]][0])],
        voting="soft",
    )
    cv = cross_val_score(clone(ensemble), train_x, train_y, scoring="average_precision",
                         cv=outer, n_jobs=-1)
    ensemble.fit(train_x, train_y)
    score = ensemble.predict_proba(holdout_x)[:, 1]
    fitted["ensemble"] = (ensemble, score)
    rows.append({
        "id": "ensemble",
        "name": f"Ensamble logística + {best_tree['name']}",
        "family": "Ensamble",
        "plain": "Promedia la probabilidad de los dos enfoques.",
        "params": {"members": [PRODUCTION_ID, best_tree["id"]]},
        "n_features": None,
        "cv": {"mean": float(cv.mean()), "std": float(cv.std()), "folds": len(cv)},
        "holdout": holdout_metrics(y_true, score),
        "seconds": round(time.perf_counter() - started, 1),
    })
    log(f"{rows[-1]['name']}: CV PR-AUC {cv.mean():.3f} ± {cv.std():.3f}")

    production = fitted[PRODUCTION_ID][1]
    for row in rows:
        if row["id"] not in (PRODUCTION_ID, "dummy"):
            row["vs_production"] = paired_bootstrap(y_true, fitted[row["id"]][1], production)
        if deployed != PRODUCTION_ID and row["id"] not in (deployed, "dummy"):
            row["vs_deployed"] = paired_bootstrap(y_true, fitted[row["id"]][1], fitted[deployed][1])

    ranked = sorted((r for r in rows if r["id"] != "dummy"), key=lambda r: -r["cv"]["mean"])
    return {
        "selection_metric": "cv_pr_auc",
        "holdout_size": len(y_true),
        "holdout_positive_rate": float(y_true.mean()),
        "accuracy_if_nobody_leaves": float(1 - y_true.mean()),
        "winner": ranked[0]["id"],
        "production": PRODUCTION_ID,
        "deployed": deployed,
        "notes": [
            "Hyperparameters tuned on the training split only (inner 5-fold randomized search).",
            (
                "CV of tuned candidates reuses the training split, so it slightly favours "
                "tuned models over the fixed production logistic."
            ),
            (
                "Holdout used once per candidate; vs_production is a paired bootstrap of the "
                "PR-AUC difference on the holdout."
            ),
        ],
        "candidates": rows,
    }
