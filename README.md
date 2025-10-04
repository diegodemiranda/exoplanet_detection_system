# 🛰️ NASA Space Apps Challenge 2025: 
## AI Exoplanet Detection System

The development of a complete exoplanet detection system using artificial intelligence represents one of the most exciting frontiers of modern computational astronomy. This project implements a robust and scalable solution that combines advanced machine learning techniques with data from major **NASA** space missions, fully meeting the requirements of the proposed challenge.

**`CNN + LSTM`** Hybrid Model Architecture: <br>
The core of the system is a hybrid model that combines `Convolutional Neural Networks (CNN)` with `bidirectional Long Short-Term Memory (LSTM)`. This architecture was specifically designed to capture both local features and long-term temporal dependencies in stellar light curves, overcoming the limitations of traditional models that process only isolated aspects of the data.

The model implements a *multi-branch* architecture that simultaneously processes three types of input: 
- a global view of the complete light curve (2001 points);
- a local view focused on the transit window (201 points);
- and auxiliary features derived from stellar parameters;

This triple approach allows the model to learn patterns at different temporal scales, from subtle long-term variations to the specific characteristics of transit events.

The global CNN branch utilizes three convolutional layers with increasing filters (16, 32, 64) for hierarchical feature extraction, followed by bidirectional LSTM layers that capture temporal dependencies in both directions of time. The local branch applies convolutions focused on the transit region, while the auxiliary branch processes stellar parameters through dense layers. The intelligent fusion of this information allows for accurate classification among confirmed exoplanets, candidates, and false positives.

---

## Table of Contents
- [Overview and Goals](#overview-and-goals)
- [System Architecture](#system-architecture)
- [Design Patterns and Engineering Practices](#design-patterns-and-engineering-practices)
- [Data Models and Validation](#data-models-and-validation)
- [API Endpoints and Contracts](#api-endpoints-and-contracts)
- [Caching Strategy](#caching-strategy)
- [Error Handling Strategy](#error-handling-strategy)
- [Configuration and Environment](#configuration-and-environment)
- [Containerization and Deployment](#containerization-and-deployment)
- [Security and Robustness](#security-and-robustness)
- [Monitoring and Observability Stack](#monitoring-and-observability-stack)
- [Code Quality and Tooling](#code-quality-and-tooling)
- [Libraries and Rationale](#libraries-and-rationale-selected)
- [Frontend (Web UI)](#frontend-web-ui)
- [How to Run (Dev and Docker)](#how-to-run)
- [Model File and Evaluation Guide](#model-file-and-evaluation-guide)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements and License](#acknowledgements-and-license)

---

## Overview and Goals
This repository contains a production-grade, end-to-end exoplanet detection platform designed for reliability, performance, and maintainability. It exposes:
- A FastAPI-powered inference API with strict validation and structured error handling.
- A hybrid CNN + BiLSTM model implementation in TensorFlow/Keras.
- A fully instrumented runtime with Prometheus metrics and Grafana dashboards.
- A responsive frontend SPA (served by the API).
- A hardened reverse proxy (Nginx) with HTTPS support.
- A Celery worker for background/batch tasks leveraging Redis.

---

## System Architecture
Core components:
- API Layer (FastAPI): request validation, routing, error handling, metrics, UI serving.
- Model Service (TensorFlow/Keras): hybrid CNN + BiLSTM multi-branch inference.
- Preprocessing & Feature Extraction: robust normalization, outlier clipping, local window extraction, statistical features.
- Caching: in-memory LRU + TTL (`OptimizedCache`) for prediction results and models.
- Background Worker: Celery tasks (single/batch prediction, warmup, cache maintenance, metrics collection).
- Reverse Proxy: Nginx with TLS (dev certs via Makefile) and HTTP→HTTPS redirect.
- Monitoring: Prometheus (scrape /metrics) and Grafana (pre-provisioned datasource and dashboard).
- Frontend: single-page app served at `/` with assets `/style.css` and `/app.js` from `frontend/`.

Data flow (inference):
1) Client submits a candidate with light-curve flux and mission → API validates via Pydantic V2.
2) Service preprocesses flux (median+MAD normalization, clipping, resize) and derives local window and auxiliary features.
3) TensorFlow model runs inference on [global_view, local_view, auxiliary_features].
4) API returns class prediction, probability distribution, timestamps, and quality metrics.
5) Metrics and cache are updated; Prometheus/Grafana visualize performance and reliability.

---

## Design Patterns and Engineering Practices
- Dependency Injection: lightweight service container (`ServiceContainer`) for detector service.
- Strategy Interface: `IExoplanetDetector` to enable future alternative backends (e.g., Torch) without API changes.
- Middleware: HTTP metrics collection and structured error handling.
- Background Tasks: FastAPI background tasks for logging and periodic cache cleanup via lifespan.
- Async & Concurrency: `asyncio` for non-blocking endpoints and batch/pipeline steps; semaphore-limited parallelism in batch prediction.
- Configuration as Code: centralized `config.py` with `pydantic-settings`.

---

## Data Models and Validation
Pydantic V2 models in `models.py`:
- Light Curve: min length, finite values validation, allowed missions (Kepler, K2, TESS).
- Stellar and Transit Parameters: physical ranges (e.g., Teff, logg, [Fe/H], radius, mass).
- Candidate: target_name sanitized; nested models; extras forbidden.
- PredictionResult: bounds on probabilities and confidence; timestamps and metadata.
- Batch Request/Response: size limits and batch-level statistics.

Strict validation blocks malformed inputs early, returning detailed error codes using custom exceptions (`exceptions.py`).

---

## API Endpoints and Contracts
- GET `/` → serves frontend SPA (`frontend/index.html`).
- GET `/style.css` → serves CSS from `frontend/style.css`.
- GET `/app.js` → serves JS from `frontend/app.js`.
- GET `/health` → liveness/readiness; 200 healthy or 503 if model not loaded.
- GET `/metrics` → Prometheus exposition format (latency histograms, counters, uptime).
- GET `/metrics/json` → JSON metrics (API, model, caches, system info).
- GET `/examples` → example payloads for quick manual tests.
- GET `/cache/stats` → cache stats snapshots.
- DELETE `/cache/clear` → clear prediction and model caches.
- GET `/model/info` → model loaded status, weights path, input shapes, class labels.
- POST `/predict` → single prediction (Pydantic-validated input, structured output).
- POST `/predict/batch` → batch prediction with aggregated stats.

Error codes:
- 422 for invalid input; 400/500 for processing/prediction errors; 503 when model is not ready (no weights loaded).

---

## Caching Strategy
`cache.py` implements an async LRU + TTL cache with per-entry metadata:
- TTL-based expiry and active cleanup.
- LRU eviction under pressure.
- Stats: size, hit rate, access counters, entry ages.
Prediction responses are cached using a strong key derived from candidate content (flux hash + mission + target).

---

## Error Handling Strategy
Custom exception hierarchy (`ExoplanetDetectionError` and specializations) normalized by FastAPI exception handlers:
- InvalidDataError, ModelNotLoadedError, ProcessingError, PredictionError mapped to structured JSON with codes and details.
- Logs include error context without leaking internals.

---

## Configuration and Environment
`config.py` centralizes settings via `pydantic-settings` (env + .env):
- API: title, description, version, host/port, reload.
- CORS: origins, methods, headers.
- Model: `model_path`, `model_full_path` (default `models/exoplanet_model.keras`), lengths, features, classes.
- Cache: `cache_ttl`, `max_cache_size`.
- Performance: switches for model/prediction caching; batch size.
- Logging: level/format.

Environment variables you’ll commonly set:
- `MODEL_FULL_PATH=/app/models/exoplanet_model.keras`
- `LOG_LEVEL=INFO`
- `CACHE_TTL=3600`
- `REDIS_URL=redis://redis:6379/0`

---

## Containerization and Deployment
- Dockerfile: multi-stage build (builder venv + slim runtime), BLAS/LAPACK, curl for healthcheck, non-root user, Gunicorn + Uvicorn workers.
- docker-compose.yml: full stack up by default:
  - exoplanet-api (FastAPI + SPA)
  - redis (broker/cache)
  - prometheus (scrapes `/metrics`)
  - grafana (provisioned datasource and dashboards)
  - nginx (HTTPS reverse proxy; 80→443 redirect)
  - worker (Celery tasks; calls API)
- Volumes: logs/, models/, cache/, frontend/ mounted into the API container (frontend as read-only for live HTML/CSS/JS edits without rebuild).
- Nginx TLS:
  - Generate dev certs: `make cert-dev` (creates `nginx/ssl/cert.pem` and `nginx/ssl/key.pem`).
  - Replace with production certificates in `nginx/ssl/` for real deployments.

---

## Security and Robustness
- HTTPS by default via Nginx (dev certs; swap in real certs in prod).
- Strict validation on all inputs; fields bounded and types enforced.
- Structured error responses, no internal stack traces exposed.
- CORS configurable; consider restricting in production.
- Health endpoint for readiness checks; graceful startup/shutdown with cleanup.
- Non-root user inside the container.

---

## Monitoring and Observability Stack
- Prometheus metrics exposed at `/metrics`:
  - `http_requests_total{method,path,status}`
  - `http_request_duration_seconds_bucket/sum/count` (histogram)
  - `http_errors_total{method,path,status}`
  - `predictions_total{endpoint}` and `prediction_duration_seconds` (histogram)
  - `app_uptime_seconds`
- Prometheus scrape config included: `monitoring/prometheus.yml` (targets `exoplanet-api:8000`).
- Grafana provisioning:
  - Datasource: `monitoring/grafana/datasources/datasource.yml` (Prometheus).
  - Dashboards: provider at `monitoring/grafana/provisioning/dashboards/dashboards.yml`.
  - Sample dashboard: `monitoring/grafana/dashboards/exoplanet_api_overview.json` (requests, errors, p95 latency, prediction latency, requests by path).

---

## Code Quality and Tooling
- Tests: `pytest`, `pytest-asyncio`, `pytest-cov`.
- Linting/Formatting/Types: `flake8`, `black`, `mypy`.
- Pre-commit hooks recommended.
- CI-friendly commands can be added as Make targets.

---

## Libraries and Rationale (selected)
- FastAPI, Uvicorn, Pydantic V2: modern, fast, typed API development.
- NumPy, SciPy, scikit-learn: scientific stack for preprocessing and features.
- TensorFlow/Keras: CNN + BiLSTM hybrid model inference.
- prometheus-client: first-class runtime metrics.
- Celery + Redis: background jobs, batch orchestration.
- Nginx: TLS termination, reverse proxying.
- Grafana + Prometheus: monitoring and dashboards.

(Note: PyTorch and Transformers were intentionally removed from requirements to keep the runtime focused on the Keras model currently implemented.)

---

## Frontend (Web UI)
- Files served from `frontend/` only (no root duplicates).
  - `/` → `frontend/index.html`
  - `/style.css` → `frontend/style.css`
  - `/app.js` → `frontend/app.js`
- You can live-edit `frontend/` files; the API will serve them without container rebuild.

---

## How to Run
### Development (local Python)
```bash
pip install -r requirements.txt
# Optional: create a .env with overrides (MODEL_FULL_PATH, LOG_LEVEL, etc.)
python main.py
```
- UI: http://localhost:8000/
- API docs: http://localhost:8000/docs

### Full Stack with Docker
```bash
# Optional for HTTPS via Nginx in dev
make cert-dev

# Build and start everything
docker compose up --build -d

# Tail API logs
docker compose logs -f exoplanet-api
```
- HTTPS (Nginx): https://localhost/
- API: http://localhost:8000/
- Prometheus: http://localhost:9090/
- Grafana: http://localhost:3000 (admin / admin123)

Apple Silicon note (M1/M2): if TensorFlow wheels fail, build/run as amd64:
```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build --no-cache
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up -d
```

---

## Model File and Evaluation Guide
- The API expects a full Keras model at `models/exoplanet_model.keras` (configurable via `MODEL_FULL_PATH`).
- If the model file is not present, the API loads and serves the UI/metrics but returns `503 Service Unavailable` on `/health` and denies predictions with a clear error message.
- When the model file is available:
  1) Place the file under `./models/exoplanet_model.keras` (or set `MODEL_FULL_PATH`).
  2) Restart the API; check `/model/info` (`loaded:true`).
  3) POST `/predict` with a validated candidate payload.

### Measuring Efficiency
- Runtime latency: use `/metrics` histograms (`http_request_duration_seconds`, `prediction_duration_seconds`) to compute p50/p95/p99.
- Throughput and error rates: counters `http_requests_total`, `http_errors_total`, `predictions_total`.
- Use load tools (e.g., `hey`, `wrk`, `k6`) to benchmark predictions while Prometheus records time series.
- Evaluate data quality: the service computes a `quality_score` (0..1) from signal statistics to aid interpretability.

---

## Troubleshooting
- “Could not open requirements file … /tmp/requirements.txt”: the Dockerfile now installs from `/app/requirements.txt`; rebuild without cache:
  ```bash
  docker compose down -v
  docker builder prune -f
  docker compose build --no-cache exoplanet-api worker
  docker compose up -d
  ```
- `/health` returns 503: model file missing or path misconfigured → place `exoplanet_model.keras` under `./models` or set `MODEL_FULL_PATH`.
- Nginx TLS errors: run `make cert-dev` (dev) or place real certs in `nginx/ssl/`.
- Port 8000 busy: stop local processes (macOS: `lsof -nP -iTCP:8000 -sTCP:LISTEN`).
- Prometheus not scraping: ensure `exoplanet-api` is up, and `/metrics` returns data; check `monitoring/prometheus.yml` target.

---

## Acknowledgements and License
- NASA Kepler/K2/TESS communities and datasets
- FastAPI, Pydantic, NumPy/SciPy, TensorFlow/Keras
- Prometheus and Grafana projects

License: MIT (see LICENSE).
