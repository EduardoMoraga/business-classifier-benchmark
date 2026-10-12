import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_dataset import load_attrition_csv
from src.pipeline import run_pipeline


def main() -> None:
    df = load_attrition_csv()
    summary = run_pipeline(df, Path("."))
    best = max(summary["metrics"], key=lambda row: row["pr_auc"])
    print(f"Dataset rows: {summary['rows']}")
    print(f"Best model by PR-AUC: {best['model']} ({best['pr_auc']:.3f})")
    print("Wrote app/public/*.json, reports/*.csv, and models/*.joblib")


if __name__ == "__main__":
    main()
