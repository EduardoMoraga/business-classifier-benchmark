"""Run the model tournament on the same split as the production pipeline.

    uv run python scripts/run_tournament.py --case ibm
    uv run python scripts/run_tournament.py --case hr

Writes app/public/<case>/tournament.json (read by the web app) and reports/<case>/tournament.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cases import CASES, load_case
from src.models.tournament import ORIGINAL_ID, PRODUCTION_ID, run_tournament
from src.pipeline import XGBOOST_PARAMS, prepare_dataset, split_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(CASES), default="ibm")
    case = CASES[parser.parse_args().case]

    df, _ = load_case(case, PROJECT_ROOT)
    x, y = prepare_dataset(df, case.drop_columns)
    train_x, test_x, train_y, test_y = split_dataset(x, y)
    original = case.production == "xgboost"
    result = run_tournament(
        train_x, train_y, test_x, test_y,
        features=case.features,
        original_xgboost=XGBOOST_PARAMS if original else None,
        deployed=ORIGINAL_ID if original else PRODUCTION_ID,
        reference_label="referencia" if original else "actual",
    )
    result["case"] = case.id
    result["run_at"] = datetime.now(UTC).isoformat()

    public = PROJECT_ROOT / "app" / "public" / case.id / "tournament.json"
    public.write_text(json.dumps(result, indent=2))
    table = pd.DataFrame(
        {
            "model": r["name"],
            "cv_pr_auc": r["cv"]["mean"],
            "cv_sd": r["cv"]["std"],
            **{f"holdout_{k}": v for k, v in r["holdout"].items() if k != "confusion"},
        }
        for r in result["candidates"]
    )
    table.to_csv(PROJECT_ROOT / "reports" / case.id / "tournament.csv", index=False)
    print(f"[{case.id}] Winner by CV PR-AUC: {result['winner']}. Wrote {public}")


if __name__ == "__main__":
    main()
