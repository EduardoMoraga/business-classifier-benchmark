from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/ibm_hr_attrition.csv")
HR_RAW_PATH = Path("data/raw/hr_empleados.xlsx")
HR_SOURCE_URL = (
    "https://github.com/EduardoMoraga/employee-attrition-predictor/raw/main/data/empleados.xlsx"
)

# The Spanish workbook is a translation of Kaggle's HR_comma_sep.csv; restore the original
# English names so the typed-decision models read the same vocabulary as in the IBM case.
HR_COLUMNS = {
    "satisfaccion": "satisfaction_level",
    "evaluacion": "last_evaluation",
    "numero_proyecto": "number_project",
    "horas_mensuales": "average_monthly_hours",
    "antiguedad_empresa": "time_spend_company",
    "accidentes": "work_accident",
    "promocionado": "promotion_last_5years",
    "departamento": "department",
    "nivel_salario": "salary",
    "abandono": "Attrition",
}
HR_VALUES = {
    "department": {
        "ventas": "sales", "tecnico": "technical", "soporte": "support", "IT": "IT",
        "producto": "product_mng", "marketing": "marketing", "laboratorio": "RandD",
        "contabilidad": "accounting", "recursos_humanos": "hr", "gerencia": "management",
    },
    "salary": {"bajo": "low", "intermedio": "medium", "alto": "high"},
    "Attrition": {"Si": "Yes", "No": "No"},
}
HR_NUMERIC = [
    "satisfaction_level", "last_evaluation", "number_project", "average_monthly_hours",
    "time_spend_company", "work_accident", "promotion_last_5years",
]


def _missing(path: Path, hint: str) -> FileNotFoundError:
    return FileNotFoundError(f"Missing {path}. {hint} See docs/dataset-card.md.")


def load_attrition_csv(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise _missing(path, "Download the IBM HR Analytics CSV from Kaggle.")
    return pd.read_csv(path)


def parse_hr_workbook(raw: pd.DataFrame) -> pd.DataFrame:
    """The workbook stores each CSV line in a single cell; split it and normalise names."""
    column = raw.columns[0]
    df = raw[column].str.split(",", expand=True)
    df.columns = [name.strip() for name in column.split(",")]
    df = df.apply(lambda s: s.str.strip()).rename(columns=HR_COLUMNS)
    for name, mapping in HR_VALUES.items():
        df[name] = df[name].map(mapping)
    df[HR_NUMERIC] = df[HR_NUMERIC].apply(pd.to_numeric)
    int_columns = ["number_project", "average_monthly_hours", "time_spend_company",
                   "work_accident", "promotion_last_5years"]
    df[int_columns] = df[int_columns].astype(int)
    return df


def load_hr_workbook(path: Path = HR_RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise _missing(path, f"Download it with: curl -L -o {path} {HR_SOURCE_URL}")
    return parse_hr_workbook(pd.read_excel(path))
