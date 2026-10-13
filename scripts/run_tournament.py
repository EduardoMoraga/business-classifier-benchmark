"""Run the model tournament on the same split as the production pipeline.

    uv run python scripts/run_tournament.py

Writes app/public/tournament.json (read by the web app) and reports/tournament.csv.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_dataset import RAW_PATH, load_attrition_csv
from src.models.tournament import run_tournament
from src.pipeline import TARGET, prepare_dataset, split_dataset


def main() -> None:
    x, y = prepare_dataset(load_attrition_csv(PROJECT_ROOT / RAW_PATH))
    train_x, test_x, train_y, test_y = split_dataset(x, y)
    result = run_tournament(train_x, train_y, test_x, test_y, TARGET)
    result["run_at"] = datetime.now(UTC).isoformat()

    public = PROJECT_ROOT / "app" / "public" / "tournament.json"
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
    table.to_csv(PROJECT_ROOT / "reports" / "tournament.csv", index=False)
    print(f"Winner by CV PR-AUC: {result['winner']}. Wrote {public}")


if __name__ == "__main__":
    main()
