from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Any, Optional
import logging
from datetime import datetime
import asyncio

# Optimized local imports
from config import settings
from models import (
    OptimizedExoplanetCandidate,
    OptimizedPredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from services import OptimizedExoplanetDetectorService
from exceptions import (
    ExoplanetDetectionError,
    ModelNotLoadedError,
    InvalidDataError,
    ProcessingError,
    PredictionError
)
from cache import prediction_cache, model_cache

# New: data ingestion utilities and mounting Dash
from data_ingestion import fetch_catalogs, _load_flux_from_local_cache, _download_light_curve
from starlette.middleware.wsgi import WSGIMiddleware

# Clear existing metrics from registry to avoid duplication
def clear_prometheus_registry():
    """Remove all metrics from the registry to avoid duplication"""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

# Clear registry before creating metrics
clear_prometheus_registry()

# Prometheus metrics definitions
REQUEST_COUNT = Counter(
    'http_requests_total', 'Total HTTP requests', ['method', 'path', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 'HTTP request latency in seconds', ['method', 'path']
)
ERROR_COUNT = Counter(
    'http_errors_total', 'Total HTTP errors', ['method', 'path', 'status']
)
PREDICTIONS_TOTAL = Counter(
    'predictions_total', 'Total predictions made', ['endpoint']
)
PREDICTION_LATENCY = Histogram(
    'prediction_duration_seconds', 'Prediction latency in seconds', ['endpoint']
)
UPTIME_SECONDS = Gauge('app_uptime_seconds', 'Application uptime in seconds')

# Optimized logging configuration
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('exoplanet_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Service container (Dependency Injection)
class ServiceContainer:
    """Container for dependency injection"""

    def __init__(self):
        self._services = {}

    def register(self, name: str, service: Any):
        """Register a service"""
        self._services[name] = service

    def get(self, name: str):
        """Get a registered service"""
        if name not in self._services:
            raise ValueError(f"Serviço '{name}' não registrado")
        return self._services[name]

# Global container instance
container = ServiceContainer()

# Dependency functions
async def get_detector_service() -> OptimizedExoplanetDetectorService:
    """Dependency to obtain the detector service"""
    try:
        return container.get("detector_service")
    except ValueError:
        raise ModelNotLoadedError()

# Custom middleware for metrics
class MetricsMiddleware:
    """Middleware for metrics collection"""

    def __init__(self, app: FastAPI):
        self.app = app
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.now()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            self.request_count += 1

            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    if status_code >= 400:
                        self.error_count += 1
                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application's lifecycle in an optimized way"""
    # Startup
    logger.info("🚀 Initializing Exoplanet Detection API...")
    cleanup_task = None

    try:
        # Initialize services
        detector_service = OptimizedExoplanetDetectorService()
        container.register("detector_service", detector_service)

        # Background cache cleanup task
        async def cleanup_caches():
            while True:
                await asyncio.sleep(3600)  # Every hour
                await prediction_cache.cleanup_expired()
                await model_cache.cleanup_expired()
                logger.info("Cache cleanup executed")

        cleanup_task = asyncio.create_task(cleanup_caches())

        logger.info("✅ Services initialized successfully!")
        yield

    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🔄 Shutting down application...")
        if cleanup_task:
            cleanup_task.cancel()
        await prediction_cache.clear()
        await model_cache.clear()
        logger.info("✅ Application finalized")

# Create optimized FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Add metrics middleware
metrics_middleware = MetricsMiddleware(app)

@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """HTTP middleware to collect metrics without breaking FastAPI pipeline"""
    metrics_middleware.request_count += 1
    start = datetime.now()
    try:
        response = await call_next(request)
        latency = (datetime.now() - start).total_seconds()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(latency)
        REQUEST_COUNT.labels(request.method, request.url.path, str(getattr(response, "status_code", 200))).inc()
        if getattr(response, "status_code", 200) >= 400:
            metrics_middleware.error_count += 1
            ERROR_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
        # Uptime gauge
        UPTIME_SECONDS.set((datetime.now() - metrics_middleware.start_time).total_seconds())
        return response
    except Exception as ex:
        metrics_middleware.error_count += 1
        ERROR_COUNT.labels(request.method, request.url.path, '500').inc()
        REQUEST_COUNT.labels(request.method, request.url.path, '500').inc()
        raise

# Custom exception handler
@app.exception_handler(ExoplanetDetectionError)
async def detection_exception_handler(request, exc: ExoplanetDetectionError):
    """Handler for exoplanet detection exceptions"""
    logger.error(f"Erro de detecção: {exc.message} - {exc.details}")
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP_ERROR", "message": exc.detail}
    )

# Optimized endpoints
@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def root():
    """Redirect to the new Dash UI"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/", status_code=307)

@app.get("/style.css", include_in_schema=False)
async def serve_css():
    return FileResponse("frontend/style.css", media_type="text/css")

@app.get("/app.js", include_in_schema=False)
async def serve_js():
    return FileResponse("frontend/app.js", media_type="application/javascript")

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Optimized health check"""
    try:
        detector = await get_detector_service()
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "version": settings.api_version,
            "services": {
                "detector": "loaded",
                "cache": "active"
            },
            "uptime_seconds": (datetime.now() - metrics_middleware.start_time).total_seconds(),
            "request_count": metrics_middleware.request_count,
            "error_rate": metrics_middleware.error_count / max(metrics_middleware.request_count, 1)
        }
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Expose metrics in Prometheus format"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics/json", tags=["Monitoring"])
async def get_metrics(detector: OptimizedExoplanetDetectorService = Depends(get_detector_service)):
    """Detailed system metrics (JSON)"""
    try:
        print("[LOG] Route /metrics/json called")
        model_metrics = detector.get_metrics()
        print(f"[LOG] model_metrics: {model_metrics}")
        response = {
            "model_metrics": model_metrics,
            "api_metrics": {
                "total_requests": metrics_middleware.request_count,
                "error_count": metrics_middleware.error_count,
                "error_rate": metrics_middleware.error_count / max(metrics_middleware.request_count, 1),
                "uptime_seconds": (datetime.now() - metrics_middleware.start_time).total_seconds()
            },
            "cache_metrics": {
                "prediction_cache": prediction_cache.stats(),
                "model_cache": model_cache.stats()
            },
            "system_info": {
                "version": settings.api_version,
                "timestamp": datetime.now()
            }
        }
        print(f"[LOG] response: {response}")
        return response
    except Exception as e:
        print(f"[LOG] Error in /metrics/json route: {e}")
        raise HTTPException(status_code=500, detail=f"Error obtaining metrics: {str(e)}")

@app.post("/predict", response_model=OptimizedPredictionResult, tags=["Prediction"])
async def predict_exoplanet(
    candidate: OptimizedExoplanetCandidate,
    background_tasks: BackgroundTasks,
    detector: OptimizedExoplanetDetectorService = Depends(get_detector_service)
):
    """
    Optimized exoplanet prediction with advanced validation
    """
    start = datetime.now()
    try:
        background_tasks.add_task(log_prediction_request, candidate.target_name)
        result = await detector.predict(candidate)
        PREDICTIONS_TOTAL.labels('single').inc()
        PREDICTION_LATENCY.labels('single').observe((datetime.now() - start).total_seconds())
        return result
    except InvalidDataError as e:
        raise HTTPException(status_code=422, detail=e.message)
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=e.message)

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    detector: OptimizedExoplanetDetectorService = Depends(get_detector_service)
):
    """Optimized batch prediction"""
    start_time = datetime.now()

    try:
        background_tasks.add_task(
            log_batch_request,
            len(request.candidates),
            [c.target_name for c in request.candidates]
        )
        results = await detector.predict_batch(request.candidates)
        total_time = (datetime.now() - start_time).total_seconds()
        PREDICTIONS_TOTAL.labels('batch').inc()
        PREDICTION_LATENCY.labels('batch').observe(total_time)

        # Batch statistics
        batch_stats = {
            "total_candidates": len(request.candidates),
            "successful_predictions": len(results),
            "failed_predictions": len(request.candidates) - len(results),
            "success_rate": len(results) / len(request.candidates),
            "avg_processing_time": total_time / len(request.candidates),
            "predictions_by_class": {}
        }

        # Count predictions by class
        for result in results:
            class_name = result.prediction
            batch_stats["predictions_by_class"][class_name] = batch_stats["predictions_by_class"].get(class_name, 0) + 1

        return BatchPredictionResponse(
            results=results,
            total_processing_time=total_time,
            batch_stats=batch_stats,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {str(e)}")

@app.get("/examples", tags=["Utilities"])
async def get_examples():
    """Optimized examples for testing"""
    return {
        "confirmed_planet": {
            "target_name": "Kepler-452b",
            "light_curve": {
                "flux": [1.0002, 0.9998, 0.9995, 1.0001, 0.9997, 0.9985, 0.9990, 1.0003] * 250,
                "mission": "Kepler"
            },
            "stellar_params": {
                "teff": 5757,
                "logg": 4.32,
                "feh": 0.21,
                "radius": 1.11,
                "mass": 1.04
            }
        },
        "candidate_planet": {
            "target_name": "TOI-1234.01",
            "light_curve": {
                "flux": [1.0001, 0.9999, 0.9996, 1.0002, 0.9998, 0.9992, 0.9995, 1.0001] * 250,
                "mission": "TESS"
            }
        },
        "false_positive": {
            "target_name": "Random-Star-001",
            "light_curve": {
                "flux": [1.0 + 0.001 * (i % 10 - 5) for i in range(2000)],
                "mission": "K2"
            }
        }
    }

@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """Detailed cache statistics"""
    return {
        "prediction_cache": prediction_cache.stats(),
        "model_cache": model_cache.stats(),
        "timestamp": datetime.now()
    }

@app.delete("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear cache (admin only)"""
    await prediction_cache.clear()
    await model_cache.clear()
    return {"message": "Cache cleared successfully", "timestamp": datetime.now()}

@app.get("/model/info", tags=["Monitoring"])
async def model_info(detector: OptimizedExoplanetDetectorService = Depends(get_detector_service)):
    """Information about the loaded model for troubleshooting"""
    return detector.get_model_info()

# New: Catalog search endpoint (for Dash UI)
@app.get("/catalog/search", tags=["Catalog"])
async def search_catalog(
    query: Optional[str] = Query(default="", description="Search by target or star name"),
    mission: Optional[List[str]] = Query(default=None, description="Filter by mission (repeat param for multiple)"),
    status: Optional[List[str]] = Query(default=None, description="Filter by status: CONFIRMED, CANDIDATE, FALSE_POSITIVE"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
):
    """Search exoplanet catalog with basic filters and pagination.
    Uses a cached copy per mission selection to keep responses fast.
    """
    try:
        missions = mission or ["Kepler", "TESS", "K2"]
        cache_key = f"catalog:{','.join(sorted(missions))}"
        records = await model_cache.get(cache_key)
        if records is None:
            # Run potentially blocking fetch in a worker thread
            records = await asyncio.to_thread(fetch_catalogs, missions)
            await model_cache.set(cache_key, records, ttl=1800)  # 30 minutes

        # Apply filters
        q = (query or "").strip().lower()
        filtered = []
        for r in records:
            if q and q not in str(r.get("target_name", "")).lower():
                continue
            if status:
                if (r.get("label") or "").upper() not in {s.upper() for s in status}:
                    continue
            filtered.append(r)

        total = len(filtered)

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        page_items = filtered[start:end]

        # Normalize output fields for the UI table
        items = [
            {
                "target_name": it.get("target_name"),
                "mission": it.get("mission"),
                "status": (it.get("label") or "").upper(),
                "ids": it.get("ids", {}),
                # Placeholders for table columns we may not have yet
                "star_name": None,
                "orbital_period": None,
                "planet_radius": None,
                "distance_ly": None,
            }
            for it in page_items
        ]
        return {"total": total, "items": items, "page": page, "page_size": page_size}
    except Exception as e:
        logger.exception(f"Error in /catalog/search: {e}")
        raise HTTPException(status_code=500, detail="Catalog search failed")

# New: Light curve retrieval endpoint
@app.get("/lightcurve", tags=["Catalog"])
async def get_lightcurve(
    mission: str = Query(..., description="Mission: Kepler, TESS, or K2"),
    kepid: Optional[int] = Query(None),
    tic: Optional[int] = Query(None),
    epic: Optional[int] = Query(None),
    target_name: Optional[str] = Query(None),
    max_points: int = Query(2000, ge=100, le=10000),
    download: bool = Query(False, description="Allow remote download if not in local cache")
):
    """Return light curve data (flux and synthetic time) for a given target.
    Prefers local cache; optionally attempts remote download if allowed.
    """
    try:
        mission_up = (mission or "").title()
        ids = {"kepid": kepid, "tic": tic, "epic": epic}
        # Build a compact cache key
        id_part = next((f"{k}:{v}" for k, v in ids.items() if v is not None), target_name or "unknown")
        lc_key = f"lc:{mission_up}:{id_part}"

        cached = await model_cache.get(lc_key)
        if cached is not None:
            flux = cached
        else:
            rec = {"mission": mission_up, "ids": {k: v for k, v in ids.items() if v is not None}}
            # Try local cache first (run in thread)
            flux = await asyncio.to_thread(_load_flux_from_local_cache, rec)
            if flux is None and download:
                # Try remote download as fallback
                flux = await asyncio.to_thread(_download_light_curve, rec)
            if flux is None:
                raise HTTPException(status_code=404, detail="Light curve not found in cache")
            await model_cache.set(lc_key, flux, ttl=3600)

        import numpy as np
        arr = np.asarray(flux, dtype=float)
        n = arr.size
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points, dtype=int)
            arr = arr[idx]
            n = arr.size
        # Synthetic time axis (indices in days relative)
        time = (np.arange(n, dtype=float) / 48.0).tolist()  # assuming 30 min cadence
        return {"mission": mission_up, "time": time, "flux": arr.tolist(), "target_name": target_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /lightcurve: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve light curve")

# Helper functions for background tasks
async def log_prediction_request(target_name: str):
    """Background log for prediction request"""
    logger.info(f"Prediction requested for: {target_name}")

async def log_batch_request(count: int, target_names: List[str]):
    """Background log for batch request"""
    logger.info(f"Batch prediction requested for {count} candidates: {target_names[:5]}...")

# Development endpoint (debug mode only)
if settings.reload:
    @app.get("/dev/reload", include_in_schema=False)
    async def reload_models():
        """Reload models (development only)"""
        try:
            detector_service = OptimizedExoplanetDetectorService()
            container.register("detector_service", detector_service)
            return {"message": "Models reloaded", "timestamp": datetime.now()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Adicionar logs detalhados para depuração
logger.info("Iniciando a montagem do Dash UI...")
try:
    from dashboard.app import create_dash_app  # local module
    import flask

    # Criar o servidor Dash/Flask com o prefixo correto
    flask_server, dash_app = create_dash_app(requests_pathname_prefix="/ui/")
    app.mount("/ui", WSGIMiddleware(flask_server))
    logger.info("Dash UI montado com sucesso na rota /ui")
except Exception as e:
    logger.error(f"Erro ao montar o Dash UI: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True
    )
