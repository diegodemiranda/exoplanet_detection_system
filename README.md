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
- [Quick overview](#quick-overview)
- [What the service provides](#what-the-service-provides)
- [Run locally (Python environment)](#run-locally-python-environment)
- [Frontend (development and production build)](#frontend-development-and-production-build)
- [Run with Docker Compose (full stack)](#run-with-docker-compose-full-stack)
- [Configuration and environment](#configuration-and-environment)
- [Logs and observability](#logs-and-observability)
- [How the model is used](#how-the-model-is-used)
- [Model File and Evaluation Guide](#model-file-and-evaluation-guide)
- [Measuring Efficiency](#measuring-efficiency)
- [Troubleshooting](#troubleshooting)
- [Developer notes and where to look in the code](#developer-notes-and-where-to-look-in-the-code)
- [Acknowledgements and License](#acknowledgements-and-license)

---
## Quick overview

- Backend: FastAPI application at `src/backend/main.py` (run with Uvicorn: `backend.main:app`).
- Frontend: React + Vite in `src/frontend` (dev with `npm run dev`, build with `npm run build` → output `src/frontend/dist/`).
- Model: expected Keras file `models/exoplanet_model.keras` (configurable via environment variables).
- Containers & services: `docker-compose.yml` defines the full stack used in development (API, Redis, Prometheus, Grafana, Nginx, worker).
- Logs: main API log file: `exoplanet_api.log`.


## What the service provides

A FastAPI-based HTTP API exposing (at least) the following routes:

- `GET /` → serves the frontend `dist/index.html` when the frontend is built; otherwise returns a small HTML landing page with build instructions.
- `GET /health` → health & readiness information (returns `503` when the model is not loaded).
- `GET /metrics` → Prometheus exposition format for runtime metrics.
- `GET /metrics/json` → JSON-formatted metrics (requires the detector service to be available).
- `POST /predict` → single prediction endpoint (expects validated candidate payload).
- `POST /predict/batch` → batch prediction endpoint.
- `GET /examples` → example payloads for quick manual tests.

> Note: The backend implements structured exception handling and custom metrics (see `src/backend/main.py`). When the model is missing or not loaded, prediction endpoints will return an error and `/health` will reflect model readiness.


## Run locally (Python environment)

Recommended: use a virtualenv. The repository also contains `start.sh` which attempts to use a Conda environment named `nasa_project`.

Using a virtualenv and Uvicorn (recommended if you don't use conda):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ensure PYTHONPATH points to the `src` directory so package-style imports work
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# start the FastAPI app in development mode
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Using the included helper script (`start.sh`):

```bash
chmod +x start.sh
./start.sh
```

> `start.sh` sets `PYTHONPATH` to `src/` and tries to activate a Conda environment named `nasa_project` before running `uvicorn`.


## Frontend (development and production build)

The frontend is a React app powered by Vite located at `src/frontend`.

**Development server**

```bash
cd src/frontend
npm install    # or yarn
npm run dev
# the Vite dev server will show the port (commonly http://localhost:5173)
```

**Production build**

```bash
cd src/frontend
npm run build
# the static files will be emitted to src/frontend/dist/
# the backend root route (GET /) serves frontend/dist/index.html when present
```


## Run with Docker Compose (full stack)

The repository includes `docker-compose.yml` that assembles API, Redis, Prometheus, Grafana, Nginx and worker services. This is the easiest way to launch the full stack locally (including observability).

```bash
# optionally create dev TLS certs used by nginx
make cert-dev

# build and start everything in detached mode
docker compose up --build -d

# tail API logs
docker compose logs -f exoplanet-api
```

If you run on Apple Silicon and encounter binary wheel issues (TensorFlow, etc.), consider building the containers for amd64:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build --no-cache
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up -d
```


## Configuration and environment

The service uses `src/backend/config.py` for application settings. Noteworthy environment variables:

- `MODEL_FULL_PATH` — path to the Keras model file (default: `models/exoplanet_model.keras`).
- `LOG_LEVEL` — logging level (e.g. `INFO`, `DEBUG`).
- `CACHE_TTL` — time-to-live for caches (seconds).

If the model file is missing or not loadable, the API will still serve health/metrics and the frontend, but prediction endpoints will fail and `/health` will return `503` until the model is available.


## Logs and observability

- API logs are written to `exoplanet_api.log` (in addition to stdout).
- Prometheus metrics exposed at `/metrics` and a JSON summary at `/metrics/json`.
- Grafana provisioning files are in `monitoring/grafana` and a sample dashboard is provided.


## How the model is used

The project expects a Keras model file at `models/exoplanet_model.keras`. When present, the backend loads a detector service that performs preprocessing and prediction. If you do not have the model and want to run the service for UI/metrics only, the API will run but prediction endpoints will return service-unavailable errors until a model is provided.


## Model File and Evaluation Guide

- The API expects a full Keras model at `models/exoplanet_model.keras` (configurable via `MODEL_FULL_PATH`).
- If the model file is not present, the API loads and serves the UI/metrics but returns `503 Service Unavailable` on `/health` and denies predictions with a clear error message.
- When the model file is available:
  1. Place the file under `./models/exoplanet_model.keras` (or set `MODEL_FULL_PATH`).
  2. Restart the API; check `/model/info` (`loaded:true`).
  3. `POST /predict` with a validated candidate payload.


## Measuring Efficiency

- Runtime latency: use `/metrics` histograms (`http_request_duration_seconds`, `prediction_duration_seconds`) to compute p50/p95/p99.
- Throughput and error rates: counters `http_requests_total`, `http_errors_total`, `predictions_total`.
- Use load tools (e.g., `hey`, `wrk`, `k6`) to benchmark predictions while Prometheus records time series.
- Evaluate data quality: the service computes a `quality_score` (0..1) from signal statistics to aid interpretability.


## Troubleshooting

- `/health` returns 503: model file missing or path misconfigured → place `exoplanet_model.keras` under `./models` or set `MODEL_FULL_PATH`.
- Dash UI not loading assets: ensure you are visiting `/ui/` and that the app is running on the same origin; set `EXO_API_BASE_URL` if using a separate domain.
- Prometheus not scraping: ensure `exoplanet-api` is up, and `/metrics` returns data; check `monitoring/prometheus.yml` target.
- Port 8000 busy: stop local processes (macOS):

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```


## Developer notes and where to look in the code

- Backend entrypoint and routes: `src/backend/main.py`.
- Detector implementation and model-loading logic: `src/backend/exoplanet_detector_model.py` and `src/backend/services.py`.
- Data models and validation: `src/backend/models.py`.
- Cache implementation: `src/backend/cache.py`.
- Frontend sources: `src/frontend/src/` and top-level React components in that folder.
- Monitoring: `monitoring/prometheus.yml` and `monitoring/grafana`.


---

## Acknowledgements and License

- NASA Kepler/K2/TESS communities and datasets
- FastAPI, Pydantic, NumPy/SciPy, TensorFlow/Keras, Dash/Plotly
- Prometheus and Grafana projects

License: MIT (see LICENSE).
