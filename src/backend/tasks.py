import os
from celery import Celery
import httpx

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

app = Celery("exoplanet_tasks", broker=REDIS_URL, backend=REDIS_URL)

@app.task
def ping():
    return "pong"

@app.task
def predict_single(candidate: dict):
    """Execute a single prediction via API and return the JSON result"""
    url = f"{API_BASE_URL}/predict"
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, json=candidate)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e), "endpoint": url}

@app.task
def predict_batch(candidates: list):
    """Execute a batch prediction via API and return the JSON result"""
    url = f"{API_BASE_URL}/predict/batch"
    payload = {"candidates": candidates}
    try:
        with httpx.Client(timeout=300.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e), "endpoint": url}

@app.task
def warmup_model():
    """Force service initialization and attempt to warm up the model"""
    health_url = f"{API_BASE_URL}/health"
    examples_url = f"{API_BASE_URL}/examples"
    try:
        with httpx.Client(timeout=30.0) as client:
            h = client.get(health_url)
            # If not healthy, try a predict with a lightweight example
            ex = client.get(examples_url).json()
            example = ex.get("confirmed_planet") or next(iter(ex.values()), None)
            if example:
                candidate = {
                    "target_name": example["target_name"],
                    "light_curve": example["light_curve"],
                }
                client.post(f"{API_BASE_URL}/predict", json=candidate)
            return {"health_status": h.status_code}
    except Exception as e:
        return {"error": str(e)}

@app.task
def clear_cache():
    """Request API cache cleanup"""
    url = f"{API_BASE_URL}/cache/clear"
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.delete(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e), "endpoint": url}

@app.task
def fetch_metrics():
    """Obtain metrics in JSON from the API for aggregation"""
    url = f"{API_BASE_URL}/metrics/json"
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e), "endpoint": url}
