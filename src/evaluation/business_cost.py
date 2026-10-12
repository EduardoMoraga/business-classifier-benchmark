def expected_attrition_cost(
    y_true,
    y_score,
    threshold: float,
    false_negative_cost: float = 10000.0,
    false_positive_cost: float = 1000.0,
) -> float:
    """Estimate simple intervention cost for a threshold.

    False negative: missed employee who leaves.
    False positive: unnecessary retention intervention.
    """
    total = 0.0
    for actual, score in zip(y_true, y_score):
        predicted = score >= threshold
        if actual and not predicted:
            total += false_negative_cost
        elif not actual and predicted:
            total += false_positive_cost
    return total
