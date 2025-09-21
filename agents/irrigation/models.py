# server/agents/irrigation/models.py
"""
Pydantic models for irrigation agent
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import date as dt_date

class IrrigationRequest(BaseModel):
    lat: float = Field(..., description="Latitude of the location")
    lon: float = Field(..., description="Longitude of the location")
    timezone: str = Field("auto", description="Timezone for the location")
    crop: str = Field(..., description="Crop type (e.g., wheat, rice, cotton)")
    sowing_date: dt_date = Field(..., description="Date when crop was sown")
    soil_texture: Literal["sand", "sandy_loam", "loam", "clay_loam", "clay"] = Field("loam", description="Soil texture type")
    elevation_m: float = Field(100.0, description="Elevation in meters")
    planning_horizon_days: int = Field(10, ge=1, le=21, description="Number of days to plan ahead")
    rain_skip_mm: float = Field(5.0, ge=0, description="Minimum rain amount to skip irrigation (mm)")
    target_depth_mm: float = Field(20.0, ge=5, le=80, description="Target irrigation depth (mm)")

class DailyWeather(BaseModel):
    date: Optional[dt_date] = None
    tmin_c: Optional[float] = None
    tmax_c: Optional[float] = None
    rh_mean: Optional[float] = None
    wind_u2_ms: Optional[float] = None
    precip_mm: Optional[float] = None
    rs_mj_m2: Optional[float] = None

class IrrigationEvent(BaseModel):
    date: str
    mm: float
    reason: str
    confidence: str

class RainEvent(BaseModel):
    date: str
    mm: float

class ETCData(BaseModel):
    date: str
    kc: Optional[float]
    etc: Optional[float]

class ET0Data(BaseModel):
    date: str
    et0: Optional[float]

class IrrigationPlan(BaseModel):
    irrigation_events: List[IrrigationEvent]
    rain_events: List[RainEvent]
    et0_daily: List[ET0Data]
    etc_daily: List[ETCData]
    deficit_end_mm: float
    total_irrigation_mm: float
    next_irrigation_date: Optional[str]

class IrrigationResponse(BaseModel):
    success: bool
    data: IrrigationPlan
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None