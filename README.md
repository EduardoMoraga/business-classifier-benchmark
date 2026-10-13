# Business Classifier Benchmark

Portfolio-grade data science project comparing:

1. A parsimonious classical ML classifier.
2. Jev / TypeSafe System One typed decisions.
3. Laya typed-decision classification.

Primary business use case: employee attrition classification using IBM HR Analytics Employee Attrition & Performance.

## Current status

The project is runnable end to end for the classical model and portfolio web app.

Verified local run:

- Dataset rows: 1,470
- Classical holdout PR-AUC: 0.591
- Classical holdout ROC-AUC: 0.809
- Repeated CV PR-AUC: 0.618 ± 0.052 across 25 folds
- Web build: passing
- Python tests: passing
- npm audit: 0 vulnerabilities

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

Run the Python pipeline:

```bash
uv run python scripts/run_pipeline.py
```

This writes:

- `app/public/model-summary.json` (metrics, train-only EDA, segments, curves, calibration, threshold sweep)
- `app/public/holdout.json` (all holdout rows with classical score and typed state)
- `app/public/holdout-sample.json` (first 12 rows, kept for compatibility)
- `reports/model_metrics.csv`
- `reports/logistic_coefficients.csv`
- `reports/permutation_importance.csv`
- `models/logistic_attrition.joblib`

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
uv run python scripts/benchmark_typed_models.py --provider jev --limit 20
uv run python scripts/benchmark_typed_models.py --provider laya   # full holdout
```

Outputs:

- `reports/{run}_typed_predictions.json`: per-row score, latency and error (failed rows are reported, never imputed).
- `app/public/typed-benchmark.json`: provider metrics plus the classical model on the exact same scored rows, a 50/50 blend with the classical score, and a paired bootstrap of the PR-AUC difference. The web app picks it up automatically.

Hybrid variant (LLM as a deployment layer): the same profile plus the logistic score, its training percentile and the employee's top risk drivers.

```bash
uv run python scripts/benchmark_typed_models.py --provider jev --variant hybrid
uv run python scripts/benchmark_typed_models.py --provider jev --variant hybrid --from-report  # re-summarize, no API calls
```

## Model tournament

```bash
uv run python scripts/run_tournament.py
```

Tunes logistic L2, Random Forest, XGBoost, LightGBM, Boruta + logistic, Boruta + XGBoost and a soft-voting ensemble on the training split only (inner 5-fold randomized search), scores each with RepeatedStratifiedKFold 5x5, evaluates once on the holdout and compares against the production logistic with a paired bootstrap. Writes `app/public/tournament.json` and `reports/tournament.csv`. XGBoost on macOS needs `brew install libomp`.

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
src/data/            Dataset loading
src/features/        Preprocessing, VIF, correlation reports
src/models/          Classical and typed-state helpers
src/integrations/    Jev and Laya clients
src/evaluation/      Metrics and business cost helpers
scripts/             Pipeline and typed-model benchmark runners
supabase/            SQL schema for raw/curated/ml layers
docs/                Brief, methodology, design, dataset card
```
