"""
Validadores de dados otimizados com Pydantic V2
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np
from datetime import datetime

from config import settings
from exceptions import InvalidDataError


class OptimizedLightCurveData(BaseModel):
    """Modelo otimizado para dados de curva de luz com validações avançadas"""
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    flux: List[float] = Field(..., description="Dados de fluxo normalizado", min_length=100)
    time: Optional[List[float]] = Field(None, description="Timestamps correspondentes")
    mission: str = Field("Kepler", description="Missão espacial")

    @field_validator('flux')
    @classmethod
    def validate_flux(cls, v):
        """Validação avançada dos dados de fluxo"""
        if len(v) > 10000:
            raise InvalidDataError(
                field="flux",
                value=len(v),
                expected="máximo 10000 pontos"
            )

        # Verifica valores finitos
        if not all(np.isfinite(val) for val in v):
            raise InvalidDataError(
                field="flux",
                expected="todos os valores devem ser finitos (não NaN ou inf)"
            )

        return v

    @field_validator('mission')
    @classmethod
    def validate_mission(cls, v):
        """Validação da missão"""
        valid_missions = ["Kepler", "K2", "TESS"]
        if v not in valid_missions:
            raise InvalidDataError(
                field="mission",
                value=v,
                expected=f"uma das missões: {valid_missions}"
            )
        return v


class OptimizedStellarParameters(BaseModel):
    """Parâmetros estelares com validações físicas"""
    model_config = ConfigDict(extra="forbid")

    teff: Optional[float] = Field(None, ge=2000, le=50000, description="Temperatura efetiva (K)")
    logg: Optional[float] = Field(None, ge=0, le=6, description="log g (cm/s²)")
    feh: Optional[float] = Field(None, ge=-3, le=1, description="Metalicidade [Fe/H]")
    radius: Optional[float] = Field(None, gt=0, le=100, description="Raio estelar (R_sun)")
    mass: Optional[float] = Field(None, gt=0, le=50, description="Massa estelar (M_sun)")


class OptimizedTransitParameters(BaseModel):
    """Parâmetros de trânsito com validações físicas"""
    model_config = ConfigDict(extra="forbid")

    period: Optional[float] = Field(None, gt=0, le=10000, description="Período orbital (dias)")
    epoch: Optional[float] = Field(None, description="Época do trânsito (BJD)")
    duration: Optional[float] = Field(None, gt=0, le=24, description="Duração do trânsito (horas)")
    depth: Optional[float] = Field(None, gt=0, le=1, description="Profundidade do trânsito (fração)")


class OptimizedExoplanetCandidate(BaseModel):
    """Modelo otimizado para candidato a exoplaneta"""
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    target_name: str = Field(..., min_length=1, max_length=100, description="Nome do alvo")
    light_curve: OptimizedLightCurveData
    stellar_params: Optional[OptimizedStellarParameters] = None
    transit_params: Optional[OptimizedTransitParameters] = None

    @field_validator('target_name')
    @classmethod
    def validate_target_name(cls, v):
        """Validação do nome do alvo"""
        cleaned = v.strip()
        if not cleaned:
            raise InvalidDataError(field="target_name", expected="nome não vazio")
        return cleaned


class OptimizedPredictionResult(BaseModel):
    """Resultado otimizado de predição com mais metadados"""
    model_config = ConfigDict(extra="forbid")

    target_name: str
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float]
    processing_time: float = Field(..., ge=0)
    timestamp: datetime
    version: str = Field(default="1.0.0")
    features_used: List[str] = Field(default_factory=list)
    quality_score: Optional[float] = Field(None, ge=0, le=1)

    @field_validator('prediction')
    @classmethod
    def validate_prediction(cls, v):
        """Validação da predição"""
        valid_predictions = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
        if v not in valid_predictions:
            raise InvalidDataError(
                field="prediction",
                value=v,
                expected=f"uma das opções: {valid_predictions}"
            )
        return v


class BatchPredictionRequest(BaseModel):
    """Requisição para predição em lote"""
    model_config = ConfigDict(extra="forbid")

    candidates: List[OptimizedExoplanetCandidate] = Field(..., min_length=1, max_length=32)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchPredictionResponse(BaseModel):
    """Resposta para predição em lote"""
    model_config = ConfigDict(extra="forbid")

    results: List[OptimizedPredictionResult]
    total_processing_time: float
    batch_stats: Dict[str, Any]
    timestamp: datetime
