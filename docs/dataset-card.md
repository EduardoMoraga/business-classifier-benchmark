# Dataset card

## Dataset

IBM HR Analytics Employee Attrition & Performance.

## Source

- Kaggle listing: <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset>
- Working raw CSV mirror used for local development: <https://raw.githubusercontent.com/IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance/main/WA_Fn-UseC_-HR-Employee-Attrition.csv>

## Shape

- Rows: 1,470
- Columns: 35 in raw file
- Target: `Attrition`
- Class balance: 237 yes / 1,233 no

## License and publication risk

License must be verified from Kaggle before public release. Until then, do not commit the raw CSV to GitHub. The repo keeps `data/raw/` ignored and publishes only derived model summaries.

## Known risks

- Dataset is synthetic/fictitious, so conclusions are methodological rather than operational.
- Dataset is common online, so hosted AI tools may have seen it before.
- The positive class is imbalanced; PR-AUC is the primary metric.
- `EmployeeCount`, `EmployeeNumber`, `Over18`, and `StandardHours` are dropped from modeling because they are identifiers or constants.

---

# Case 2: HR Analytics (employee-attrition-predictor)

## Source

- Workbook from the earlier portfolio project: <https://github.com/EduardoMoraga/employee-attrition-predictor> (`data/empleados.xlsx`, SHA-256 `4c00e485…3d903c792`).
- It is a Spanish translation of Kaggle's HR Analytics `HR_comma_sep.csv`. `src/data/load_dataset.py` restores the original English column names and values.
- Download: `curl -L -o data/raw/hr_empleados.xlsx https://github.com/EduardoMoraga/employee-attrition-predictor/raw/main/data/empleados.xlsx`

## Shape

- Rows: 14,999 raw; 11,991 after dropping 3,008 exact duplicate rows
- Columns: 10 (9 features + `Attrition`)
- Class balance after dedupe: 16.6% yes (raw file: 23.8%)

## Known risks

- 20% of the raw rows are exact copies. A random split leaves 31% of test rows identical to a training row, which inflates any metric; duplicates are dropped before the split.
- The data shows the clean, near-separable structure typical of simulated data; use it for method, not for business conclusions.
- `satisfaction_level` is a survey answer whose timing relative to the exit is unknown, so it may partly reflect the decision to leave.
