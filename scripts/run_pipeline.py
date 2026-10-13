"""Run the production pipeline for one case (or both).

    uv run python scripts/run_pipeline.py             # every case
    uv run python scripts/run_pipeline.py --case hr
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cases import CASES, load_case
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    args = parser.parse_args()
    for case in CASES.values() if args.case == "all" else [CASES[args.case]]:
        df, duplicates = load_case(case, PROJECT_ROOT)
        summary = run_pipeline(df, PROJECT_ROOT, case, duplicates)
        production = summary["metrics"][-1]
        print(f"[{case.id}] {summary['rows']} rows ({duplicates} duplicates removed) · "
              f"{production['model']} holdout PR-AUC {production['pr_auc']:.3f} · "
              f"CV {summary['cv']['mean']:.3f} ± {summary['cv']['std']:.3f}")
    print("Wrote app/public/<case>/*.json, reports/<case>/*.csv and models/*.joblib")


if __name__ == "__main__":
    main()
