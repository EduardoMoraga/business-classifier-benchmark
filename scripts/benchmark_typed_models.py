"""Benchmark Jev and Laya on exported holdout states.

Run the classical pipeline first:

    uv run python scripts/run_pipeline.py

Then configure .env and call one provider:

    uv run python scripts/benchmark_typed_models.py --provider laya --limit 20
    uv run python scripts/benchmark_typed_models.py --provider jev --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _score_from_response(response: dict) -> float | None:
    value = response.get("attrition_risk")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("noul", "probability", "score"):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
    return None


def _classify(provider: str, state: str) -> dict:
    if provider == "jev":
        from src.integrations.jev_client import classify_attrition_with_jev

        return classify_attrition_with_jev(state)
    if provider == "laya":
        from src.integrations.laya_client import classify_attrition_with_laya

        return classify_attrition_with_laya(state)
    raise ValueError(f"Unknown provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["jev", "laya"], required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    load_dotenv()
    sample_path = PROJECT_ROOT / "app" / "public" / "holdout-sample.json"
    rows = json.loads(sample_path.read_text())[: args.limit]

    predictions = []
    for index, row in enumerate(rows):
        response = _classify(args.provider, row["typed_state"])
        predictions.append(
            {
                "row_index": index,
                "provider": args.provider,
                "y_true": row["y_true"],
                "y_score": _score_from_response(response),
                "raw_response": response,
            }
        )

    output = PROJECT_ROOT / "reports" / f"{args.provider}_typed_predictions.json"
    output.write_text(json.dumps(predictions, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
