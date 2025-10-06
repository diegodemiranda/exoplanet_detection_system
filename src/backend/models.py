from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

# All comments and docstrings in English. No import changes needed.

class OptimizedLightCurveData(BaseModel):
    flux: List[float] = Field(..., description="Normalized flux data")
    time: Optional[List[float]] = Field(None, description="Corresponding timestamps")
    mission: str = Field("Kepler", description="Space mission")

    @field_validator('flux')
    def validate_flux(cls, v):
        if len(v) > 10000:
            raise ValueError("Flux data exceeds maximum allowed points (10,000).")
        return v

    @field_validator('mission')
    def validate_mission(cls, v):
        valid_missions = ["Kepler", "K2", "TESS"]
        if v not in valid_missions:
            raise ValueError(f"Mission must be one of {valid_missions}.")
        return v

# Updated other classes similarly...
