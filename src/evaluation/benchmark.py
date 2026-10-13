"""Score typed-decision predictions against the same holdout rows as the classical model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"pr_auc": None, "roc_auc": None, "brier": None, "f1": None}
    both_classes = len(set(y_true.tolist())) == 2
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else None,
        "brier": float(brier_score_loss(y_true, y_score)),
        "f1": float(f1_score(y_true, y_score >= 0.5, zero_division=0)),
    }


def _confusion(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    pred = y_score >= threshold
    truth = y_true.astype(bool)
    return {
        "threshold": threshold,
        "tp": int((pred & truth).sum()),
        "fp": int((pred & ~truth).sum()),
        "fn": int((~pred & truth).sum()),
        "tn": int((~pred & ~truth).sum()),
    }


def summarize_provider(predictions: list[dict], classical_scores: dict[int, float]) -> dict:
    """Metrics for one provider plus the classical model on the exact same scored rows.

    Each prediction needs row_id, y_true, y_score (None when the call failed) and latency_ms.
    """
    scored = [p for p in predictions if isinstance(p.get("y_score"), (int, float))]
    y_true = np.array([bool(p["y_true"]) for p in scored])
    typed = np.array([float(p["y_score"]) for p in scored])
    classical = np.array([float(classical_scores[p["row_id"]]) for p in scored])
    latencies = [p["latency_ms"] for p in predictions if p.get("latency_ms") is not None]

    return {
        "rows_requested": len(predictions),
        "rows_scored": len(scored),
        "positives": int(y_true.sum()) if len(y_true) else 0,
        "metrics": _metrics(y_true, typed),
        "confusion": _confusion(y_true, typed),
        "classical_same_rows": _metrics(y_true, classical),
        "classical_same_rows_confusion": _confusion(y_true, classical),
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
    }


def blend_summary(predictions: list[dict], classical_scores: dict[int, float]) -> dict:
    """Simple 50/50 average of the typed score and the classical score on the scored rows.

    No weights are fitted, so the holdout stays an honest test of the combination.
    """
    scored = [p for p in predictions if isinstance(p.get("y_score"), (int, float))]
    y_true = np.array([bool(p["y_true"]) for p in scored])
    blend = np.array([(float(p["y_score"]) + float(classical_scores[p["row_id"]])) / 2 for p in scored])
    return {"weights": {"typed": 0.5, "classical": 0.5}, "metrics": _metrics(y_true, blend),
            "confusion": _confusion(y_true, blend)}


def vs_classical(predictions: list[dict], classical_scores: dict[int, float]) -> dict | None:
    """Paired bootstrap of PR-AUC(typed) - PR-AUC(classical) on the scored rows.

    rank_agreement (Spearman) shows whether the typed model adds its own ordering or mostly
    echoes the classical score it was given.
    """
    from scipy.stats import spearmanr

    from src.models.tournament import paired_bootstrap

    scored = [p for p in predictions if isinstance(p.get("y_score"), (int, float))]
    y_true = np.array([int(bool(p["y_true"])) for p in scored])
    if len(set(y_true.tolist())) < 2:
        return None
    typed = np.array([float(p["y_score"]) for p in scored])
    classical = np.array([float(classical_scores[p["row_id"]]) for p in scored])
    return {
        **paired_bootstrap(y_true, typed, classical),
        "rank_agreement": float(spearmanr(typed, classical).statistic),
    }
