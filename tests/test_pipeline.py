import pandas as pd

from src.pipeline import DROP_COLUMNS, TARGET, prepare_dataset, run_pipeline


def test_prepare_dataset_drops_target_and_constant_identifiers():
    df = pd.DataFrame(
        {
            TARGET: ["Yes", "No"],
            "EmployeeCount": [1, 1],
            "EmployeeNumber": [10, 11],
            "Over18": ["Y", "Y"],
            "StandardHours": [80, 80],
            "Age": [30, 40],
        }
    )

    x, y = prepare_dataset(df)

    assert TARGET not in x.columns
    assert all(column not in x.columns for column in DROP_COLUMNS)
    assert y.tolist() == [1, 0]


def test_run_pipeline_exports_portfolio_artifacts(tmp_path):
    rows = []
    for i in range(80):
        rows.append(
            {
                "Age": 24 + (i % 35),
                "Attrition": "Yes" if i % 5 == 0 else "No",
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 500 + i,
                "Department": "Sales" if i % 2 else "Research & Development",
                "DistanceFromHome": 1 + (i % 20),
                "Education": 1 + (i % 5),
                "EducationField": "Life Sciences",
                "EmployeeCount": 1,
                "EmployeeNumber": i,
                "EnvironmentSatisfaction": 1 + (i % 4),
                "Gender": "Female" if i % 2 else "Male",
                "HourlyRate": 40 + (i % 30),
                "JobInvolvement": 1 + (i % 4),
                "JobLevel": 1 + (i % 5),
                "JobRole": "Sales Executive",
                "JobSatisfaction": 1 + (i % 4),
                "MaritalStatus": "Single" if i % 3 == 0 else "Married",
                "MonthlyIncome": 2500 + i * 10,
                "MonthlyRate": 10000 + i,
                "NumCompaniesWorked": i % 6,
                "Over18": "Y",
                "OverTime": "Yes" if i % 4 == 0 else "No",
                "PercentSalaryHike": 11 + (i % 10),
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 1 + (i % 4),
                "StandardHours": 80,
                "StockOptionLevel": i % 4,
                "TotalWorkingYears": 1 + (i % 20),
                "TrainingTimesLastYear": i % 5,
                "WorkLifeBalance": 1 + (i % 4),
                "YearsAtCompany": 1 + (i % 10),
                "YearsInCurrentRole": i % 7,
                "YearsSinceLastPromotion": i % 5,
                "YearsWithCurrManager": i % 7,
            }
        )

    summary = run_pipeline(pd.DataFrame(rows), tmp_path)

    assert summary["rows"] == 80
    assert (tmp_path / "app" / "public" / "model-summary.json").exists()
    assert (tmp_path / "reports" / "model_metrics.csv").exists()
    assert (tmp_path / "models" / "logistic_attrition.joblib").exists()
