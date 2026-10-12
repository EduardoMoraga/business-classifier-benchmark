import pandas as pd

from src.models.typed_state import employee_row_to_state


def test_employee_state_excludes_target():
    state = employee_row_to_state(pd.Series({"Age": 40, "Attrition": "Yes"}))
    assert "Age: 40" in state
    assert "Attrition" not in state
