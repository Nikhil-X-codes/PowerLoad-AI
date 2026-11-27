# PowerLoad AI — Power System Load Type Prediction (End-to-End MLOps)

An end-to-end, CPU-friendly, leakage-free machine learning and **MLOps** platform that classifies industrial electricity consumption into `Light_Load`, `Medium_Load`, or `Maximum_Load`, complete with a FastAPI inference service, an interactive React dashboard, automated testing, Docker containers, and CI/CD.

---

## 1. Problem Statement
Industrial power systems record 15-minute electrical measurements (`Usage_kWh`, reactive power, `CO2(tCO2)`, power factors, `NSM`, and `Date_Time`). The objective is to:
- Classify each measurement interval into `Light_Load`, `Medium_Load`, or `Maximum_Load` without future-to-past data leakage.
- Reserve the final calendar month (**December 2018**) strictly as a held-out test set and report accuracy, precision, recall, and F1-score.
- Operationalize the trained model into a production-ready full-stack application with reproducible deployment and automated CI/CD gates.

---

## 2. Solution Approach
- **Chronological & Leakage-Free Modeling:** Enforces strict temporal partitioning (`Jan–Oct` training, `Nov` validation, `Dec` held-out test) with all imputers and scalers fit exclusively on development folds.
- **Deterministic Cleaning & Feature Engineering:** Parses day-first timestamps, flags and replaces physically invalid power factors (`< 0` or `> 100`) and inconsistent `NSM` values with `NaN` + binary indicator columns, and derives cyclic time features (`hour_sin`, `hour_cos`, `ts_nsm`, `is_weekend`) and `total_reactive_power_magnitude` inside a reusable scikit-learn `Pipeline`.
- **Model Selection & Tuning:** Benchmarks 6 candidate classifiers on November validation data, tunes `RandomForest` via expanding-window `TimeSeriesSplit` + `RandomizedSearchCV` optimizing `f1_macro`, and refits on `Jan–Nov` before single-pass evaluation on December (`93.32%` accuracy, `0.9261` Macro F1).
- **Production Serving & Full-Stack Integration:** Wraps the serialized `best_model.joblib` pipeline in a centralized FastAPI inference service (`ModelInferenceService`) and connects it to the `PowerLoad AI` React/TypeScript dashboard.

---

## 3. Architecture

### Full-Stack Production Architecture
```text
User / Browser
   │
   ▼
PowerLoad AI Frontend (React + Vite + TypeScript + Tailwind)
   │  • Overview | Single Prediction | Batch Prediction | Model Performance | Explainability
   │  • REST API / HTTPS (Axios + AbortController cancellation + cold-start handling)
   ▼
FastAPI Backend (src/load_prediction/api.py)
   │  • Input Validation (Pydantic + CSV schema/size/row limits)
   │  • CORS, Structured JSON Request Logging, Exception Handling
   ▼
Centralized Inference Service (src/load_prediction/service.py)
   │  • Loads models/best_model.joblib once at startup (v1.0.0)
   │  • Existing Feature Engineering & Preprocessing Pipeline
   ▼
Trained ML Model (RandomForest) ──► Light_Load | Medium_Load | Maximum_Load + Probabilities
```

### Offline ML Training & Evaluation Pipeline
```text
data/load_data.csv (35,041 rows)
   ├──► Data Audit & Cleaning (data.py)
   ├──► Chronological Split: Jan–Oct (29,184) | Nov (2,880) | Dec (2,977)
   ├──► Development EDA (eda.py -> outputs/figures/)
   ├──► Feature Pipeline (features.py)
   ├──► Candidate Benchmarking & TimeSeriesSplit Tuning (train.py)
   └──► Held-Out Evaluation, Permutation Importance & Error Analysis (evaluate.py)
```

---

## 4. Key Features (ML & MLOps)

### MLOps & Production Engineering
- **Model Versioning & Artifact Governance:** Tracks `model_version` (`1.0.0`), hyperparameters, feature set, and class labels in [models/model_metadata.json](models/model_metadata.json) alongside [models/best_model.joblib](models/best_model.joblib).
- **Centralized Inference Service:** `ModelInferenceService` loads the fitted pipeline once at server startup (`153.89 ms` load, `193.75 MB` RSS), guarantees 100% parity with notebook predictions, and instruments per-request `inference_time_ms`.
- **Observability & Diagnostics:** Structured JSON HTTP logging, live `/api/v1/health` (model status, memory RSS, uptime), `/api/v1/metrics` (cached evaluation metrics), and `/api/v1/analytics` (data quality, model comparison, permutation importance, weekday vs. weekend drift).
- **Security Hardening:** Rejects untrusted `.joblib`/`.pkl` uploads (`415`), enforces CSV upload size (`10 MB`) and row (`50,000`) limits (`413`), restricts CORS via `ALLOWED_ORIGINS`, and validates all inputs with Pydantic (`422`).
- **Automated Testing Pyramid:** Backend unit, parity, API, security, and performance tests (`pytest`, 23 tests), frontend component tests (`vitest` + React Testing Library, 3 tests), and browser E2E spec (`playwright`).
- **Containerization & CI/CD:** Production `Dockerfile` (non-root user, slim image excluding training data and notebooks via `.dockerignore`), `docker-compose.yml` full-stack orchestration, and GitHub Actions workflow ([assignment2-ci.yml](../.github/workflows/assignment2-ci.yml)) that blocks Docker builds if any backend or frontend test fails.

### Interactive Dashboard (`PowerLoad AI`)
- **Overview:** Dataset volume (`35,041` records), held-out accuracy & Macro F1, historical monthly consumption, and class distribution.
- **Single Prediction:** Real-time classification with auto-derived `NSM`, range validation, and class probability bars.
- **Batch Prediction:** Drag-and-drop CSV upload, downloadable sample template, paginated table, load distribution summary, and CSV export.
- **Model Performance & Explainability:** Confusion matrix, per-class metrics, candidate model comparison chart, permutation importance, and error/drift analysis.

## Directory Structure
```text
Assignment 2/
├── AGENTS.md
├── README.md
├── PROGRESS.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── data/
│   └── load_data.csv
├── notebooks/
│   └── load_prediction.ipynb
├── src/
│   └── load_prediction/
│       ├── __init__.py
│       ├── data.py
│       ├── eda.py
│       ├── features.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       ├── service.py
│       └── api.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_predict.py
│   ├── test_service.py
│   ├── test_api.py
│   └── test_security_performance.py
├── frontend/                   # PowerLoad AI (React + Vite + TypeScript + Tailwind)
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── playwright.config.ts
│   ├── vitest.config.ts
│   ├── e2e/powerload.spec.ts
│   └── src/
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
└── outputs/
    ├── data_quality.json
    ├── model_comparison.csv
    ├── tuning_results.json
    ├── explainability.json
    ├── predictions.csv
    ├── metrics.json
    └── figures/
```

## Setup & Execution
1. Place `load_data.csv` at `data/load_data.csv`.
2. Install dependencies (`Python 3.13.5`, CPU-only):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the end-to-end pipeline:
   ```bash
   python -m src.load_prediction.data
   python -m src.load_prediction.eda
   python -m src.load_prediction.train
   python -m src.load_prediction.evaluate
   ```
4. Run batch inference on new CSV data:
   ```bash
   python -m src.load_prediction.predict --input data/load_data.csv --output outputs/batch_preds.csv
   ```
   > **Security note:** Never load untrusted `.joblib` files; `joblib.load` uses pickle deserialization.
5. Run automated tests:
   ```bash
   pytest tests/ -v
   ```
6. Execute the notebook from a clean kernel:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/load_prediction.ipynb
   ```

## Data Assumptions & Cleaning
- **Dataset:** `35,041` rows × `9` columns (`2018-01-01 00:00:00` to `2018-12-31 23:45:00`, parsed with `dayfirst=True`).
- **Ordering & Duplicates:** Sorted chronologically (`mergesort`) to fix 365 daily `00:00` out-of-order rows. Preserved all rows including the single duplicate timestamp (`2018-12-31 20:00:00`).
- **Validity Checks:** Replaced physically invalid Power Factors (`< 0` or `> 100`: `3,226` lagging, `2,685` leading) and timestamp-inconsistent `NSM` values (`3,414` rows) with `NaN` plus binary indicator flags. Extreme plausible `Usage_kWh` values (up to `435.02`) were preserved.
- **Imputation:** Train-fitted median imputation (`SimpleImputer(strategy="median", add_indicator=True)`) is applied strictly inside the scikit-learn pipeline after chronological splitting.

## Chronological Split
- **Train (Jan–Oct 2018):** `29,184` rows
- **Validation (Nov 2018):** `2,880` rows
- **Test (Dec 2018, held-out):** `2,977` rows

## Model Comparison & Tuning
- **Validation Benchmark (Nov):** `RandomForest` (`macro_f1=0.5944`), `XGBoost` (`0.5922`), `HistGradientBoosting` (`0.5915`), `DecisionTree` (`0.5893`), `LogisticRegression` (`0.5495`), `DummyClassifier` (`0.2147`). `LightGBM` was skipped (`No module named 'lightgbm'`).
- **Hyperparameter Tuning (`TimeSeriesSplit(n_splits=3)`, `RandomizedSearchCV`, `f1_macro`, `random_state=42`):** Winner was `RandomForest` (`combined` features, `n_estimators=250`, `max_depth=15`, `min_samples_leaf=1`, `class_weight="balanced_subsample"`) with CV Macro F1 = `0.9082`.

## Final Test Results (December 2018, 2,977 rows)
- **Accuracy:** `0.9332`
- **Macro Precision / Recall / F1:** `0.9091` / `0.9476` / `0.9261`
- **Weighted Precision / Recall / F1:** `0.9383` / `0.9332` / `0.9338`
- **Per-Class F1:** `Light_Load`: `0.9461` | `Medium_Load`: `0.9184` | `Maximum_Load`: `0.9140`

## Limitations
- **Seasonal / Holiday Drift:** Weekend error rate in December (`13.54%`) is higher than weekday error rate (`3.42%`), with `Light_Load -> Medium_Load` (`89`) and `Light_Load -> Maximum_Load` (`69`) as top confusion pairs during year-end schedule shifts.
- **Classification vs. Forecasting:** Predicts `Load_Type` from contemporaneous measurements and timestamps, not future demand forecasting.

## Figures (`outputs/figures/`)
1. [01_target_class_distribution.png](outputs/figures/01_target_class_distribution.png) — Development set class imbalance.
2. [02_hourly_monthly_usage.png](outputs/figures/02_hourly_monthly_usage.png) — Hourly and monthly `Usage_kWh` trends.
3. [03_missing_value_distribution.png](outputs/figures/03_missing_value_distribution.png) — Raw vs. post-cleaning missing/invalid counts.
4. [04_correlation_heatmap.png](outputs/figures/04_correlation_heatmap.png) — Feature correlation matrix.
5. [05_usage_by_load_type.png](outputs/figures/05_usage_by_load_type.png) — `Usage_kWh` distribution by `Load_Type`.
6. [06_feature_distributions_boxplots.png](outputs/figures/06_feature_distributions_boxplots.png) — Numeric feature boxplots and outliers.
7. [07_weekday_vs_weekend.png](outputs/figures/07_weekday_vs_weekend.png) — Weekday vs. weekend hourly consumption.
8. [08_confusion_matrix.png](outputs/figures/08_confusion_matrix.png) — December held-out test confusion matrix.
9. [09_permutation_importance.png](outputs/figures/09_permutation_importance.png) — Validation permutation importance (`f1_macro`).
10. [10_error_analysis.png](outputs/figures/10_error_analysis.png) — December error rate by hour & top confusion pairs.

## Production Inference & FastAPI Backend
- **Centralized Inference Service (`src/load_prediction/service.py`):** Loads `models/best_model.joblib` (`RandomForest`, `model_version="1.0.0"`) once at server startup (`153.89 ms`, `193.75 MB` RSS), includes full preprocessing + feature engineering, and matches `outputs/predictions.csv` (`56.52 ms` single-record, `73.12 ms` per 100-row batch).
- **Start FastAPI Backend:**
  ```powershell
  cd "Assignment 2"
  uvicorn src.load_prediction.api:app --host 127.0.0.1 --port 8001
  ```
- **Swagger Docs:** `http://127.0.0.1:8001/docs`
- **Endpoints:**
  - `GET /api/v1/health` — Backend & model health, memory RSS, load latency
  - `GET /api/v1/model/info` — Model version, parameters, classes, input schema
  - `POST /api/v1/predict` — Single-record JSON prediction with probabilities & latency
  - `POST /api/v1/predict/batch` — CSV upload batch prediction (size/row limits enforced)
  - `GET /api/v1/metrics` — Actual evaluation metrics (`outputs/metrics.json`)
  - `GET /api/v1/analytics` — Historical dataset summaries, model comparison & explainability

## PowerLoad AI Frontend Dashboard
Located in `Assignment 2/frontend/` (React + Vite + TypeScript + Tailwind CSS + React Router + Axios):
- **Pages:** `Overview`, `Single Prediction`, `Batch Prediction`, `Model Performance`, and `Explainability` — all backed by live FastAPI endpoints.
- **Run Frontend Locally (with FastAPI running on port 8001):**
  ```powershell
  cd "Assignment 2\frontend"
  npm install
  npm run dev
  ```
  Open `http://localhost:5173/` (Vite proxies `/api` requests to `http://127.0.0.1:8001`).

## Testing, Docker & CI/CD
- **Backend Tests (`23 passed`):** `pytest tests/ -v` (unit, parity, API v1, security, and performance tests).
- **Frontend Tests (`3 passed`):** `cd frontend && npm run test` (Vitest + React Testing Library) and `npm run build` (`tsc -b && vite build`).
- **Docker Compose (local full-stack):**
  ```powershell
  cd "Assignment 2"
  docker compose up --build
  ```
  Backend runs on `http://localhost:8001` and Frontend on `http://localhost:3000`.
- **CI/CD:** [.github/workflows/assignment2-ci.yml](../.github/workflows/assignment2-ci.yml) gates `docker-build` on passing backend `pytest` and frontend `vitest` + `vite build`.
