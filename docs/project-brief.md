# Project brief

## Goal

Build an end-to-end data science portfolio project that shows business judgment, classical ML discipline, and a modern comparison against typed-decision models.

## Claim bands

- Verified: Jev and Laya both expose typed questions such as choice/score/noul.
- Verified: Laya can run locally and Jev is hosted through TypeSafe docs.
- Inferred: IBM HR Attrition is the best first dataset for narrative clarity.
- Hypothesis: Typed-decision models may be competitive on tabular classification if each row is converted into a compact business-readable state.

## Core comparison

All competitors must predict the same frozen holdout rows.

- Classical ML gets structured tabular features.
- Jev and Laya get a row-to-text state plus atomic typed questions.
- Labels are never included in the state.
- Performance is evaluated with PR-AUC, ROC-AUC, F1, Brier score, calibration, latency, and cost.
