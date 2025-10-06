"""
Custom exceptions for the application
"""
from typing import Dict, Any


class ExoplanetDetectionError(Exception):
    """Base exception for exoplanet detection errors"""

    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedError(ExoplanetDetectionError):
    """Error when the model is not loaded"""

    def __init__(self, model_name: str = "detector"):
        super().__init__(
            message=f"Model '{model_name}' was not loaded",
            error_code="MODEL_NOT_LOADED",
            details={"model_name": model_name}
        )


class InvalidDataError(ExoplanetDetectionError):
    """Error for invalid input data"""

    def __init__(self, field: str, value: Any = None, expected: str = None):
        details = {"field": field}
        if value is not None:
            details["received_value"] = str(value)
        if expected:
            details["expected"] = expected

        super().__init__(
            message=f"Invalid data in field '{field}'",
            error_code="INVALID_DATA",
            details=details
        )


class ProcessingError(ExoplanetDetectionError):
    """Error during data processing"""

    def __init__(self, operation: str, reason: str = None):
        super().__init__(
            message=f"Error during {operation}" + (f": {reason}" if reason else ""),
            error_code="PROCESSING_ERROR",
            details={"operation": operation, "reason": reason}
        )


class PredictionError(ExoplanetDetectionError):
    """Error during prediction"""

    def __init__(self, target_name: str, reason: str = None):
        super().__init__(
            message=f"Error predicting for '{target_name}'" + (f": {reason}" if reason else ""),
            error_code="PREDICTION_ERROR",
            details={"target_name": target_name, "reason": reason}
        )
