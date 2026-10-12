# Methodology

1. Freeze a stratified train/test split once.
2. Perform EDA only on training data.
3. Use correlation, VIF, and domain logic to reduce redundant variables.
4. Use PCA as an exploratory lens and optional benchmark, not as the primary explainable model.
5. Train a parsimonious logistic regression as the flagship classical model.
6. Compare against Jev and Laya on the same test rows.
7. Report accuracy metrics and business-operating metrics.

Primary metric: PR-AUC, because attrition is usually imbalanced.

Secondary metrics: ROC-AUC, F1 at selected threshold, Brier score, expected business cost, latency, and cost per 1,000 classifications.
