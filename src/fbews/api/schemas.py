"""Pydantic response/request schemas for the API (PART 30)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    data_mode: str
    models_loaded: bool
    cycles_available: int
    time: str


class Location(BaseModel):
    lat: float
    lon: float
    region: str | None = None


class PredictedError(BaseModel):
    precipitation: float = Field(..., description="mm/day")
    temperature: float = Field(..., description="K")
    wind: float = Field(..., description="m/s")
    pressure: float = Field(..., description="hPa")


class GridPointResponse(BaseModel):
    valid_time: str
    forecast_cycle: str
    lead_day: int
    location: Location
    confidence: int = Field(..., ge=0, le=100,
                            description="Derived operational indicator: 100 x (1 - bust probability)")
    confidence_band: str
    bust_probability: float = Field(..., ge=0, le=1)
    predicted_error: PredictedError
    variable_bust_probability: dict[str, float] = {}
    dominant_variable: str
    regime: str
    forecast: dict
    ensemble: dict
    volatility: dict
    analogue: dict
    explanations: list[str]
    explanation_detail: dict
    uncertainty_decomposition: dict
    series: dict
    verification: dict | None = None
    data_banner: str


class InferenceRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=360)
    lead: int = Field(3, ge=1, le=10)
    cycle: str | None = Field(None, description="YYYY-MM-DD; defaults to the latest cycle")
