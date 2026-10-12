# Acceptance checks

- [ ] Dataset source and license documented.
- [ ] Train/test split is frozen and stratified.
- [ ] EDA never uses holdout labels for decisions.
- [ ] Correlation and VIF report exist.
- [ ] PCA report exists and explains why PCA is or is not used in final model.
- [ ] Classical model exports coefficients, odds ratios, metrics, and calibration.
- [ ] Jev and Laya receive identical state strings and atomic questions.
- [ ] All models are evaluated on the same holdout rows.
- [ ] Frontend shows methodology, model comparison, and a what-if demo.
- [ ] GitHub Actions build Python checks and the static site.
