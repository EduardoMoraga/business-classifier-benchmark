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
