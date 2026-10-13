"""Benchmark Jev and Laya on exported holdout states.

Run the classical pipeline first:

    uv run python scripts/run_pipeline.py

Then configure .env and call one provider (omit --limit to score the full holdout):

    uv run python scripts/benchmark_typed_models.py --provider laya --limit 20
    uv run python scripts/benchmark_typed_models.py --provider jev
    uv run python scripts/benchmark_typed_models.py --provider jev --variant hybrid
    uv run python scripts/benchmark_typed_models.py --case hr --provider laya --workers 6

The hybrid variant sends the same profile plus the production model's score and per-employee
drivers, to test the LLM as a deployment layer on top of the validated model.

Writes reports/<case>/{run}_typed_predictions.json and merges the run summary into
app/public/<case>/typed-benchmark.json under `jev`, `laya`, `jev_hybrid` or `laya_hybrid`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import blend_summary, summarize_provider, vs_classical

STATE_FIELD = {"plain": "typed_state", "hybrid": "hybrid_state"}
CASES = ("ibm", "hr")
ATTEMPTS = 3


def _score_from_response(response: dict) -> float | None:
    answers = response.get("answers") if isinstance(response.get("answers"), dict) else response
    value = answers.get("attrition_risk")
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


def _load_holdout(case: str) -> list[dict]:
    public = PROJECT_ROOT / "app" / "public" / case
    path = public / "holdout.json"
    if not path.exists():
        path = public / "holdout-sample.json"
    return json.loads(path.read_text())


def _predict(provider: str, variant: str, index: int, row: dict) -> dict:
    """One row, retried on errors; a row that still fails is reported, never imputed."""
    started = time.perf_counter()
    response, error = {}, None
    for attempt in range(ATTEMPTS):
        started = time.perf_counter()
        try:
            response, error = _classify(provider, row[STATE_FIELD[variant]]), None
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    return {
        "row_id": row.get("row_id", index),
        "provider": provider,
        "variant": variant,
        "y_true": row["y_true"],
        "y_score": _score_from_response(response),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "error": error,
        "raw_response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="ibm")
    parser.add_argument("--provider", choices=["jev", "laya"], required=True)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API calls")
    parser.add_argument("--variant", choices=list(STATE_FIELD), default="plain")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--from-report",
        action="store_true",
        help="Recompute the summary from saved predictions without calling the API",
    )
    args = parser.parse_args()
    run_id = args.provider if args.variant == "plain" else f"{args.provider}_{args.variant}"

    load_dotenv()
    holdout = _load_holdout(args.case)
    rows = holdout[: args.limit] if args.limit else holdout

    reports = PROJECT_ROOT / "reports" / args.case / f"{run_id}_typed_predictions.json"
    if args.from_report:
        predictions = json.loads(reports.read_text())
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            predictions = list(pool.map(
                lambda item: _predict(args.provider, args.variant, *item), enumerate(rows)
            ))

    if not args.from_report:
        reports.write_text(json.dumps(predictions, indent=2))

    classical = {row.get("row_id", i): row["classical_score"] for i, row in enumerate(holdout)}
    summary = summarize_provider(predictions, classical)
    summary["blend_with_classical"] = blend_summary(predictions, classical)
    summary["vs_classical"] = vs_classical(predictions, classical)
    summary["variant"] = args.variant
    summary["case"] = args.case
    summary["run_at"] = datetime.now(UTC).isoformat()

    public = PROJECT_ROOT / "app" / "public" / args.case / "typed-benchmark.json"
    merged = json.loads(public.read_text()) if public.exists() else {"providers": {}}
    merged["providers"][run_id] = summary
    public.write_text(json.dumps(merged, indent=2))

    print(f"Wrote {reports} and {public}")
    print(f"{run_id}: {summary['rows_scored']}/{summary['rows_requested']} rows scored")


if __name__ == "__main__":
    main()
