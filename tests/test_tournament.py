import numpy as np
import pandas as pd

from src.evaluation.benchmark import blend_summary, vs_classical
from src.features.boruta_selector import BorutaSelector
from src.models.tournament import holdout_metrics, paired_bootstrap
from src.pipeline import build_logistic_pipeline, hybrid_state, person_drivers


def test_boruta_keeps_signal_and_drops_noise():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=400)
    x = np.column_stack([signal, rng.normal(size=(400, 4))])
    y = (signal + rng.normal(scale=0.3, size=400) > 0).astype(int)

    selector = BorutaSelector(max_iter=30).fit(x, y)

    assert selector.support_[0]
    assert selector.transform(x).shape[1] == selector.support_.sum() < 5


def test_holdout_metrics_confusion_counts():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.6, 0.1, 0.2])

    result = holdout_metrics(y, score)

    assert result["confusion"] == {"threshold": 0.5, "tp": 1, "fp": 1, "fn": 1, "tn": 2}
    assert result["accuracy"] == 0.6


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(scale=0.3, size=300)
    noise = rng.normal(size=300)

    better = paired_bootstrap(y, good, noise, rounds=200)
    same = paired_bootstrap(y, good, good, rounds=50)

    assert better["low"] > 0 and better["share_a_wins"] == 1.0
    assert same["low"] == same["high"] == 0.0


def test_person_drivers_name_raw_values_and_hybrid_state_keeps_profile():
    x = pd.DataFrame({
        "OverTime": ["Yes", "No", "Yes", "No"] * 10,
        "Age": np.random.default_rng(2).integers(20, 60, 40),
    })
    y = pd.Series([1, 0, 1, 0] * 10)
    model = build_logistic_pipeline(x).fit(x, y)

    drivers = person_drivers(model, x.head(2), top=2)

    assert "OverTime = Yes" in drivers[0]["raises"]
    assert "OverTime = No" in drivers[1]["lowers"]
    state = hybrid_state("Employee profile. Age: 25", 0.81, 92.0, drivers[0])
    assert state.startswith("Employee profile. Age: 25")
    assert "0.81" in state and "92%" in state and "OverTime = Yes" in state


def test_blend_averages_only_scored_rows():
    predictions = [
        {"row_id": 1, "y_true": True, "y_score": 0.8},
        {"row_id": 2, "y_true": False, "y_score": 0.2},
        {"row_id": 3, "y_true": True, "y_score": None},
    ]

    blend = blend_summary(predictions, {1: 0.6, 2: 0.4, 3: 0.9})

    assert blend["confusion"] == {"threshold": 0.5, "tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert blend["metrics"]["roc_auc"] == 1.0


def test_vs_classical_flags_a_typed_model_that_echoes_the_classical_score():
    rng = np.random.default_rng(3)
    classical = {i: float(v) for i, v in enumerate(rng.random(120))}
    predictions = [
        {"row_id": i, "y_true": classical[i] + rng.normal(scale=0.2) > 0.6, "y_score": classical[i] * 0.9}
        for i in classical
    ]

    result = vs_classical(predictions, classical)

    assert result["rank_agreement"] == 1.0
    assert result["low"] == result["high"] == 0.0
