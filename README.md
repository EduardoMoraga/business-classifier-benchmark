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

Jev and Laya adapters are implemented but require fresh API keys in `.env` before paid/remote benchmarks are run.

## Safety

Do not commit API keys. Put credentials in `.env` only. A key was shared during planning; rotate it before running benchmarks.

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

- `app/public/model-summary.json`
- `app/public/holdout-sample.json`
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
uv run python scripts/benchmark_typed_models.py --provider laya --limit 20
```

Outputs are written to `reports/{provider}_typed_predictions.json`.

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
