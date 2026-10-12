import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_pred = [score >= threshold for score in y_score]
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "threshold": threshold,
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
