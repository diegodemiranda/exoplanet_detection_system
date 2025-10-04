"""
Exceções customizadas para a aplicação
"""
from typing import Optional, Dict, Any


class ExoplanetDetectionError(Exception):
    """Exceção base para erros de detecção de exoplanetas"""

    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedError(ExoplanetDetectionError):
    """Erro quando o modelo não está carregado"""

    def __init__(self, model_name: str = "detector"):
        super().__init__(
            message=f"Modelo '{model_name}' não foi carregado",
            error_code="MODEL_NOT_LOADED",
            details={"model_name": model_name}
        )


class InvalidDataError(ExoplanetDetectionError):
    """Erro para dados inválidos"""

    def __init__(self, field: str, value: Any = None, expected: str = None):
        details = {"field": field}
        if value is not None:
            details["received_value"] = str(value)
        if expected:
            details["expected"] = expected

        super().__init__(
            message=f"Dados inválidos no campo '{field}'",
            error_code="INVALID_DATA",
            details=details
        )


class ProcessingError(ExoplanetDetectionError):
    """Erro durante processamento de dados"""

    def __init__(self, operation: str, reason: str = None):
        super().__init__(
            message=f"Erro durante {operation}" + (f": {reason}" if reason else ""),
            error_code="PROCESSING_ERROR",
            details={"operation": operation, "reason": reason}
        )


class PredictionError(ExoplanetDetectionError):
    """Erro durante predição"""

    def __init__(self, target_name: str, reason: str = None):
        super().__init__(
            message=f"Erro na predição para '{target_name}'" + (f": {reason}" if reason else ""),
            error_code="PREDICTION_ERROR",
            details={"target_name": target_name, "reason": reason}
        )
