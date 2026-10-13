"""Re-run the original employee-attrition-predictor protocol and measure what it inflated.

    uv run python scripts/audit_original.py

The original notebook split the raw workbook (with its 3,008 exact duplicate rows), early-stopped
XGBoost on the test set it then reported, and cross-validated a different model on all rows.
Writes app/public/hr/audit.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cases import CASES
from src.features.engineering import hr_features
from src.pipeline import XGBOOST_PARAMS, prepare_dataset

SEED = 42


def _design(x: pd.DataFrame) -> pd.DataFrame:
    """The notebook's encoding: engineered features, ordinal salary, one-hot department."""
    out = hr_features(x)
    out["salary"] = out["salary"].map({"low": 0, "medium": 1, "high": 2})
    return pd.get_dummies(out, columns=["department"], drop_first=True).astype(float)


def _scores(model, x_test, y_test) -> dict:
    score = model.predict_proba(x_test)[:, 1]
    return {"roc_auc": float(roc_auc_score(y_test, score)),
            "pr_auc": float(average_precision_score(y_test, score))}


def main() -> None:
    case = CASES["hr"]
    raw = case.load(PROJECT_ROOT / case.raw_path)
    x, y = prepare_dataset(raw)
    design = _design(x)
    pos_weight = float((y == 0).sum() / (y == 1).sum())

    # 1. The original protocol, as written.
    x_tr, x_te, y_tr, y_te = train_test_split(design, y, test_size=0.2, random_state=SEED, stratify=y)
    keys = pd.util.hash_pandas_object(pd.concat([x, y], axis=1), index=False)
    copied = keys.loc[x_te.index].isin(set(keys.loc[x_tr.index])).to_numpy()
    early = XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=pos_weight, eval_metric="logloss",
                          early_stopping_rounds=20, random_state=SEED, n_jobs=-1)
    early.fit(x_tr, y_tr, eval_set=[(x_te, y_te)], verbose=False)
    cv = cross_val_score(
        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                      scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED),
        design, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc",
    )

    # 2. Same split, no early stopping on test: isolates the duplicate effect.
    fixed = XGBClassifier(**XGBOOST_PARAMS, scale_pos_weight=pos_weight, eval_metric="logloss",
                          random_state=SEED, n_jobs=-1).fit(x_tr, y_tr)

    clean = json.loads((PROJECT_ROOT / "app/public/hr/model-summary.json").read_text())
    production = clean["metrics"][-1]
    audit = {
        "rows_raw": len(raw),
        "duplicate_rows": int(raw.duplicated().sum()),
        "test_rows_copied_from_train": float(copied.mean()),
        "original": {**_scores(early, x_te, y_te), "best_iteration": int(early.best_iteration),
                     "cv_roc_auc": float(cv.mean()), "test_rows": len(y_te)},
        # Same fitted model, scored only on test rows it never saw a copy of.
        "original_unseen_rows": {**_scores(early, x_te[~copied], y_te[~copied]),
                                 "test_rows": int((~copied).sum())},
        "no_early_stopping": _scores(fixed, x_te, y_te),
        "clean": {"roc_auc": production["roc_auc"], "pr_auc": production["pr_auc"],
                  "cv_pr_auc": clean["cv"]["mean"], "test_rows": clean["split"]["test"]},
    }
    out = PROJECT_ROOT / "app" / "public" / "hr" / "audit.json"
    out.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
