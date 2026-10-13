# Business Classifier Benchmark

Portfolio-grade data science project comparing:

1. A parsimonious classical ML classifier.
2. Jev / TypeSafe System One typed decisions.
3. Laya typed-decision classification.

Primary business use case: employee attrition classification, run on two datasets with the same protocol:

| Case | Dataset | Rows (holdout) | Production model |
|---|---|---|---|
| `ibm` | IBM HR Analytics Employee Attrition & Performance | 1,470 (294) | Parsimonious L1 logistic |
| `hr` | HR Analytics from [employee-attrition-predictor](https://github.com/EduardoMoraga/employee-attrition-predictor), deduplicated | 11,991 (2,399) | XGBoost with the original notebook's hyperparameters |

The `hr` case redoes the earlier XGBoost + SHAP project without its leaks (3,008 duplicate rows, early stopping on the test set) and audits the difference with `scripts/audit_original.py`.

## Current status

The project is runnable end to end for the classical model and portfolio web app.

Verified local run (holdout PR-AUC; CV is RepeatedStratifiedKFold 5x5 on train):

| | IBM | HR (your exercise) |
|---|---|---|
| Logistic L1 | 0.591 (CV 0.618) | 0.437 (CV 0.417) |
| Best tree model | 0.571 LightGBM, tie | 0.969 LightGBM / Boruta + XGBoost, tie with original XGBoost 0.966 |
| Max ROC-AUC, any model | 0.82 | 0.985 |

Live web app: https://eduardomoraga.github.io/business-classifier-benchmark/ (deployed by `.github/workflows/pages.yml` on every push to `main`).

Jev and Laya runs need API keys in `.env`; their saved predictions are committed under `reports/`.

## Safety

Do not commit API keys. Put credentials in `.env` only (gitignored).

## Quick start

```bash
cp .env.example .env
uv sync --extra dev
npm install
```

Put the raw data in `data/raw/` (see `docs/dataset-card.md`), then run the Python pipeline for both cases (or one with `--case ibm|hr`):

```bash
uv run python scripts/run_pipeline.py
```

This writes, per case:

- `app/public/<case>/model-summary.json` (metrics, train-only EDA, segments, curves, calibration, threshold sweep, global drivers)
- `app/public/<case>/holdout.json` (all holdout rows with production score, typed state and hybrid state)
- `app/public/<case>/holdout-sample.json` (first 12 rows)
- `reports/<case>/model_metrics.csv`, `reports/<case>/permutation_importance.csv` (and logistic coefficients for `ibm`)
- `models/<case>_<model>.joblib`

Run the web app:

```bash
npm run dev
```

Build the static portfolio page:

```bash
npm run build
```

## Benchmark Jev or Laya

After running the classical pipeline and adding credentials to `.env`:

```bash
uv run python scripts/benchmark_typed_models.py --case ibm --provider jev --limit 20
uv run python scripts/benchmark_typed_models.py --case hr --provider laya --workers 4   # full holdout
```

Outputs:

- `reports/<case>/{run}_typed_predictions.json`: per-row score, latency and error (each row is retried twice; rows that still fail are reported, never imputed).
- `app/public/<case>/typed-benchmark.json`: provider metrics plus the production model on the exact same scored rows, a 50/50 blend with its score, and a paired bootstrap of the PR-AUC difference. The web app picks it up automatically.

Hybrid variant (LLM as a deployment layer): the same profile plus the production model's score, its training percentile and the employee's top risk drivers (logistic contributions for `ibm`, TreeSHAP for `hr`).

```bash
uv run python scripts/benchmark_typed_models.py --case hr --provider jev --variant hybrid --workers 4
uv run python scripts/benchmark_typed_models.py --case hr --provider jev --variant hybrid --from-report  # re-summarize, no API calls
```

## Model tournament

```bash
uv run python scripts/run_tournament.py --case ibm
uv run python scripts/run_tournament.py --case hr
```

Tunes logistic L2, Random Forest, XGBoost, LightGBM, Boruta + logistic, Boruta + XGBoost and a soft-voting ensemble on the training split only (inner 5-fold randomized search), scores each with RepeatedStratifiedKFold 5x5, evaluates once on the holdout and compares against the L1 logistic with a paired bootstrap. In the `hr` case it also scores the original notebook's XGBoost untuned and compares every model against it. Boruta's fitted step is cached in `.cache/boruta` across the search. Writes `app/public/<case>/tournament.json` and `reports/<case>/tournament.csv`. XGBoost on macOS needs `brew install libomp`.

## Audit of the original exercise

```bash
uv run python scripts/audit_original.py
```

Re-runs the employee-attrition-predictor protocol on the raw workbook and writes `app/public/hr/audit.json`: the original split leaves 31% of test rows copied from train (ROC-AUC 0.994); scored only on unseen rows the same model gets 0.981, and the clean pipeline 0.983.

## Project phases

1. Dataset selection and provenance check.
2. Supabase data model and ingestion.
3. EDA, correlation, VIF, PCA exploration.
4. Parsimonious classical model.
5. Jev and Laya adapters.
6. Benchmarking with one frozen test split.
7. Static portfolio web app.

## Repository map

```txt
app/                 React/Vite portfolio web app
src/cases.py         The two datasets and how each is modelled
src/data/            Dataset loading
src/features/        Preprocessing, VIF, correlation reports
src/models/          Classical and typed-state helpers
src/integrations/    Jev and Laya clients
src/evaluation/      Metrics and business cost helpers
scripts/             Pipeline and typed-model benchmark runners
supabase/            SQL schema for raw/curated/ml layers
docs/                Brief, methodology, design, dataset card
```
