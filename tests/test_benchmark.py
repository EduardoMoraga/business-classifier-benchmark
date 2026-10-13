from src.evaluation.benchmark import summarize_provider


def test_summarize_provider_compares_on_same_scored_rows():
    predictions = [
        {"row_id": 10, "y_true": True, "y_score": 0.9, "latency_ms": 100},
        {"row_id": 11, "y_true": False, "y_score": 0.2, "latency_ms": 120},
        {"row_id": 12, "y_true": True, "y_score": None, "latency_ms": 90},
        {"row_id": 13, "y_true": False, "y_score": 0.4, "latency_ms": 110},
    ]
    classical = {10: 0.3, 11: 0.6, 12: 0.8, 13: 0.1}

    summary = summarize_provider(predictions, classical)

    assert summary["rows_requested"] == 4
    assert summary["rows_scored"] == 3
    assert summary["positives"] == 1
    assert summary["metrics"]["roc_auc"] == 1.0
    assert summary["classical_same_rows"]["roc_auc"] == 0.5
    assert summary["latency_ms_p50"] == 105.0
    assert summary["confusion"] == {"threshold": 0.5, "tp": 1, "fp": 0, "fn": 0, "tn": 2}
    assert summary["classical_same_rows_confusion"]["fp"] == 1


def test_summarize_provider_handles_no_scored_rows():
    predictions = [{"row_id": 1, "y_true": True, "y_score": None, "latency_ms": None}]

    summary = summarize_provider(predictions, {1: 0.5})

    assert summary["rows_scored"] == 0
    assert summary["metrics"]["pr_auc"] is None
    assert summary["latency_ms_p50"] is None


def test_score_parser_reads_nested_laya_answers():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_typed_models.py"
    spec = importlib.util.spec_from_file_location("bench", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    laya = {"answers": {"attrition_risk": {"type": "noul", "noul": 0.7363}}}
    assert bench._score_from_response(laya) == 0.7363
    assert bench._score_from_response({"attrition_risk": 0.2}) == 0.2
    assert bench._score_from_response({}) is None
