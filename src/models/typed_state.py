import pandas as pd


def employee_row_to_state(row: pd.Series) -> str:
    """Convert one employee row into a compact state for typed-decision models.

    Do not include the target label.
    """
    parts = []
    for key, value in row.items():
        if key.lower() == "attrition":
            continue
        parts.append(f"{key}: {value}")
    return "Employee profile. " + "; ".join(parts)
