"""
Centralized application settings
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings using Pydantic Settings"""

    # API Settings
    api_title: str = "Exoplanet Detection API"
    api_description: str = "API for exoplanet detection using AI/ML"
    api_version: str = "1.0.0"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # CORS Settings
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]

    # Model Settings (use only full .keras model)
    model_path: str = "models/"
    model_full_path: str = "models/exoplanet_model.keras"
    sequence_length: int = 2001
    local_view_length: int = 201
    n_features: int = 1
    n_classes: int = 3

    # Cache Settings
    cache_ttl: int = 3600  # 1 hour in seconds
    max_cache_size: int = 1000

    # Processing Settings
    max_flux_length: int = 10000
    min_flux_length: int = 100
    noise_threshold: float = 0.1

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Performance
    enable_model_caching: bool = True
    enable_prediction_caching: bool = True
    batch_prediction_size: int = 32

    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)


# Global settings instance
settings = Settings()
