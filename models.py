"""
Optimized data validators with Pydantic V2
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np
from datetime import datetime

from config import settings
from exceptions import InvalidDataError


class OptimizedLightCurveData(BaseModel):
    """Optimized model for light curve data with advanced validations"""
    flux: List[float] = Field(..., description="Normalized flux data", min_items=100)
    time: Optional[List[float]] = Field(None, description="Corresponding timestamps")
    mission: str = Field("Kepler", description="Space mission")

    @validator('flux')
    def validate_flux(cls, v):
        """Advanced validation for flux data"""
        if len(v) > 10000:
            raise InvalidDataError(
                field="flux",
                value=len(v),
                expected="maximum 10000 points"
            )

        # Check finite values
        if not all(np.isfinite(val) for val in v):
            raise InvalidDataError(
                field="flux",
                expected="all values must be finite (no NaN or inf)"
            )

        return v

    @validator('mission')
    def validate_mission(cls, v):
        """Validate mission name"""
        valid_missions = ["Kepler", "K2", "TESS"]
        if v not in valid_missions:
            raise InvalidDataError(
                field="mission",
                value=v,
                expected=f"one of the missions: {valid_missions}"
            )
        return v

    class Config:
        validate_assignment = True
        extra = "forbid"


class OptimizedStellarParameters(BaseModel):
    """Stellar parameters with physical validations"""
    teff: Optional[float] = Field(None, ge=2000, le=50000, description="Effective temperature (K)")
    logg: Optional[float] = Field(None, ge=0, le=6, description="log g (cm/s²)")
    feh: Optional[float] = Field(None, ge=-3, le=1, description="Metallicity [Fe/H]")
    radius: Optional[float] = Field(None, gt=0, le=100, description="Stellar radius (R_sun)")
    mass: Optional[float] = Field(None, gt=0, le=50, description="Stellar mass (M_sun)")

    class Config:
        extra = "forbid"


class OptimizedTransitParameters(BaseModel):
    """Transit parameters with physical validations"""
    period: Optional[float] = Field(None, gt=0, le=10000, description="Orbital period (days)")
    epoch: Optional[float] = Field(None, description="Epoch of the transit (BJD)")
    duration: Optional[float] = Field(None, gt=0, le=24, description="Duration of the transit (hours)")
    depth: Optional[float] = Field(None, gt=0, le=1, description="Transit depth (fraction)")

    class Config:
        extra = "forbid"


class OptimizedExoplanetCandidate(BaseModel):
    """Optimized model for exoplanet candidate"""
    target_name: str = Field(..., min_length=1, max_length=100, description="Target name")
    light_curve: OptimizedLightCurveData
    stellar_params: Optional[OptimizedStellarParameters] = None
    transit_params: Optional[OptimizedTransitParameters] = None

    @validator('target_name')
    def validate_target_name(cls, v):
        """Validate target name"""
        cleaned = v.strip()
        if not cleaned:
            raise InvalidDataError(field="target_name", expected="non-empty name")
        return cleaned

    class Config:
        validate_assignment = True
        extra = "forbid"


class OptimizedPredictionResult(BaseModel):
    """Optimized prediction result with more metadata"""
    target_name: str
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float]
    processing_time: float = Field(..., ge=0)
    timestamp: datetime
    model_version: str = Field(default="1.0.0")
    features_used: List[str] = Field(default_factory=list)
    quality_score: Optional[float] = Field(None, ge=0, le=1)

    @validator('prediction')
    def validate_prediction(cls, v):
        """Validate prediction label"""
        valid_predictions = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
        if v not in valid_predictions:
            raise InvalidDataError(
                field="prediction",
                value=v,
                expected=f"one of the options: {valid_predictions}"
            )
        return v

    class Config:
        extra = "forbid"


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    candidates: List[OptimizedExoplanetCandidate] = Field(..., min_items=1, max_items=32)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    results: List[OptimizedPredictionResult]
    total_processing_time: float
    batch_stats: Dict[str, Any]
    timestamp: datetime

    class Config:
        extra = "forbid"
