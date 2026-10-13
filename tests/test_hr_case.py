import numpy as np
import pandas as pd

from src.cases import CASES
from src.data.load_dataset import parse_hr_workbook
from src.features.engineering import hr_features
from src.pipeline import build_xgboost_pipeline, global_drivers, person_drivers, run_pipeline

HEADER = (" satisfaccion , evaluacion ,numero_proyecto, horas_mensuales , antiguedad_empresa ,"
          "accidentes,promocionado,departamento,nivel_salario,abandono")


def _workbook(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(n):
        sat = rng.random()
        leaves = sat < 0.3
        lines.append(
            f"{sat:.2f},{rng.random():.2f},{rng.integers(2, 8)},{rng.integers(96, 311)}.00,"
            f"{rng.integers(2, 11)}.00,{i % 7 == 0:d},0,{['ventas', 'IT'][i % 2]},"
            f"{['bajo', 'alto'][i % 2]},{'Si' if leaves else 'No'}"
        )
    lines += lines[:20]  # exact duplicates, as in the real workbook
    return pd.DataFrame({HEADER: lines})


def test_parse_hr_workbook_restores_english_names_and_types():
    df = parse_hr_workbook(_workbook(10))

    assert list(df.columns)[-1] == "Attrition"
    assert set(df["department"]) <= {"sales", "IT"}
    assert set(df["salary"]) <= {"low", "high"}
    assert set(df["Attrition"]) <= {"Yes", "No"}
    assert df["average_monthly_hours"].dtype.kind == "i"
    assert df["satisfaction_level"].dtype.kind == "f"


def test_hr_features_match_the_original_notebook():
    x = pd.DataFrame({"satisfaction_level": [0.5], "last_evaluation": [0.8],
                      "number_project": [0], "average_monthly_hours": [250]})

    out = hr_features(x)

    assert out["satisfaction_x_evaluation"].iloc[0] == 0.4
    assert out["hours_per_project"].iloc[0] == 250  # projects clipped at 1
    assert out["overwork"].iloc[0] == 1
    assert "overwork" not in x.columns


def test_xgboost_drivers_name_raw_and_engineered_values():
    df = parse_hr_workbook(_workbook())
    x, y = df.drop(columns="Attrition"), (df["Attrition"] == "Yes").astype(int)
    model = build_xgboost_pipeline(x, y, hr_features).fit(x, y)

    drivers = person_drivers(model, x.head(5))
    top = global_drivers(model, x)

    assert all(d["raises"] or d["lowers"] for d in drivers)
    assert any(d.startswith("satisfaction") for row in drivers for d in row["raises"] + row["lowers"])
    assert top[0]["feature"] in {"num__satisfaction_level", "num__satisfaction_x_evaluation"}
    assert top[0]["direction"] < 0  # higher satisfaction, lower risk


def test_hr_case_dedupes_and_exports_under_its_own_folder(tmp_path):
    raw = parse_hr_workbook(_workbook())
    clean = raw.drop_duplicates()

    summary = run_pipeline(clean, tmp_path, CASES["hr"], len(raw) - len(clean))

    assert summary["duplicates_removed"] == 20
    assert summary["production"] == "xgboost" and summary["top_coefficients"] == []
    assert (tmp_path / "app" / "public" / "hr" / "holdout.json").exists()
    assert (tmp_path / "models" / "hr_xgboost.joblib").exists()
