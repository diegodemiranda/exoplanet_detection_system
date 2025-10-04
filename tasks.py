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
    """Executa predição única via API e retorna o resultado JSON"""
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
    """Executa predição em lote via API e retorna o resultado JSON"""
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
    """Força inicialização do serviço e tentativa de aquecimento do modelo"""
    health_url = f"{API_BASE_URL}/health"
    examples_url = f"{API_BASE_URL}/examples"
    try:
        with httpx.Client(timeout=30.0) as client:
            h = client.get(health_url)
            # Se não estiver healthy, tentar um predict com exemplo leve
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
    """Solicita limpeza do cache da API"""
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
    """Obtém métricas em JSON da API para agregação"""
    url = f"{API_BASE_URL}/metrics/json"
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e), "endpoint": url}
