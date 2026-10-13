"""Feature engineering from the original employee-attrition-predictor notebook."""

import pandas as pd

OVERWORK_HOURS = 240


def hr_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["satisfaction_x_evaluation"] = out["satisfaction_level"] * out["last_evaluation"]
    out["hours_per_project"] = out["average_monthly_hours"] / out["number_project"].clip(lower=1)
    out["overwork"] = (out["average_monthly_hours"] > OVERWORK_HOURS).astype(int)
    return out
