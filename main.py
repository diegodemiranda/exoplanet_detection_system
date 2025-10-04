from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Any
import logging
from datetime import datetime
import asyncio

# Imports locais otimizados
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

# Limpar métricas existentes do registro para evitar duplicação
def clear_prometheus_registry():
    """Remove todas as métricas do registro para evitar duplicação"""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

# Limpar registro antes de criar métricas
clear_prometheus_registry()

# Definições de métricas Prometheus
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

# Configuração de logging otimizada
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('exoplanet_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Container de serviços (Dependency Injection)
class ServiceContainer:
    """Container para injeção de dependências"""

    def __init__(self):
        self._services = {}

    def register(self, name: str, service: Any):
        """Registra um serviço"""
        self._services[name] = service

    def get(self, name: str):
        """Obtém um serviço"""
        if name not in self._services:
            raise ValueError(f"Serviço '{name}' não registrado")
        return self._services[name]

# Instância global do container
container = ServiceContainer()

# Dependency functions
async def get_detector_service() -> OptimizedExoplanetDetectorService:
    """Dependência para obter o serviço de detecção"""
    try:
        return container.get("detector_service")
    except ValueError:
        raise ModelNotLoadedError()

# Middleware personalizado para métricas
class MetricsMiddleware:
    """Middleware para coleta de métricas"""

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
    """Gerencia o ciclo de vida da aplicação de forma otimizada"""
    # Startup
    logger.info("🚀 Inicializando Exoplanet Detection API...")
    cleanup_task = None

    try:
        # Inicializar serviços
        detector_service = OptimizedExoplanetDetectorService()
        container.register("detector_service", detector_service)

        # Task de limpeza de cache em background
        async def cleanup_caches():
            while True:
                await asyncio.sleep(3600)  # A cada hora
                await prediction_cache.cleanup_expired()
                await model_cache.cleanup_expired()
                logger.info("Cache cleanup executado")

        cleanup_task = asyncio.create_task(cleanup_caches())

        logger.info("✅ Serviços inicializados com sucesso!")
        yield

    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🔄 Finalizando aplicação...")
        if cleanup_task:
            cleanup_task.cancel()
        await prediction_cache.clear()
        await model_cache.clear()
        logger.info("✅ Aplicação finalizada")

# Criar aplicação FastAPI otimizada
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configurar CORS de forma segura
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Adicionar middleware de métricas
metrics_middleware = MetricsMiddleware(app)

@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Middleware HTTP para coletar métricas sem quebrar a pipeline do FastAPI"""
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

# Handler de exceções personalizado
@app.exception_handler(ExoplanetDetectionError)
async def detection_exception_handler(request, exc: ExoplanetDetectionError):
    """Handler para exceções de detecção"""
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
    """Handler para exceções HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP_ERROR", "message": exc.detail}
    )

# Endpoints otimizados
@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def root():
    """Página inicial servindo a interface SPA"""
    return FileResponse("frontend/index.html")

@app.get("/style.css", include_in_schema=False)
async def serve_css():
    return FileResponse("frontend/style.css", media_type="text/css")

@app.get("/app.js", include_in_schema=False)
async def serve_js():
    return FileResponse("frontend/app.js", media_type="application/javascript")

@app.get("/health", tags=["Monitoramento"])
async def health_check():
    """Verificação de saúde otimizada"""
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
        raise HTTPException(status_code=503, detail="Serviço indisponível")

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Exposição de métricas no formato Prometheus"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics/json", tags=["Monitoramento"])
async def get_metrics(detector: OptimizedExoplanetDetectorService = Depends(get_detector_service)):
    """Métricas detalhadas do sistema (JSON)"""
    try:
        model_metrics = detector.get_metrics()

        return {
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter métricas: {str(e)}")

@app.post("/predict", response_model=OptimizedPredictionResult, tags=["Predição"])
async def predict_exoplanet(
    candidate: OptimizedExoplanetCandidate,
    background_tasks: BackgroundTasks,
    detector: OptimizedExoplanetDetectorService = Depends(get_detector_service)
):
    """
    Predição otimizada de exoplaneta com validação avançada
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

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predição"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    detector: OptimizedExoplanetDetectorService = Depends(get_detector_service)
):
    """Predição em lote otimizada"""
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

        # Estatísticas do lote
        batch_stats = {
            "total_candidates": len(request.candidates),
            "successful_predictions": len(results),
            "failed_predictions": len(request.candidates) - len(results),
            "success_rate": len(results) / len(request.candidates),
            "avg_processing_time": total_time / len(request.candidates),
            "predictions_by_class": {}
        }

        # Contar predições por classe
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

@app.get("/examples", tags=["Utilitários"])
async def get_examples():
    """Exemplos otimizados para teste"""
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
    """Estatísticas detalhadas do cache"""
    return {
        "prediction_cache": prediction_cache.stats(),
        "model_cache": model_cache.stats(),
        "timestamp": datetime.now()
    }

@app.delete("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Limpa o cache (apenas para administradores)"""
    await prediction_cache.clear()
    await model_cache.clear()
    return {"message": "Cache limpo com sucesso", "timestamp": datetime.now()}

@app.get("/model/info", tags=["Monitoramento"])
async def model_info(detector: OptimizedExoplanetDetectorService = Depends(get_detector_service)):
    """Informações do modelo carregado para troubleshooting"""
    return detector.get_model_info()

# Funções auxiliares para background tasks
async def log_prediction_request(target_name: str):
    """Log de requisição de predição em background"""
    logger.info(f"Predição solicitada para: {target_name}")

async def log_batch_request(count: int, target_names: List[str]):
    """Log de requisição em lote em background"""
    logger.info(f"Predição em lote solicitada para {count} candidatos: {target_names[:5]}...")

# Endpoint de desenvolvimento (apenas em modo debug)
if settings.reload:
    @app.get("/dev/reload", include_in_schema=False)
    async def reload_models():
        """Recarrega modelos (apenas desenvolvimento)"""
        try:
            detector_service = OptimizedExoplanetDetectorService()
            container.register("detector_service", detector_service)
            return {"message": "Modelos recarregados", "timestamp": datetime.now()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True
    )
