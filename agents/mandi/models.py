# server/agents/mandi/models.py
"""
Pydantic models for mandi agent - Updated to remove default values
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MandiRequest(BaseModel):
    state: str = Field(..., description="State name")
    district: Optional[str] = Field(None, description="District name")
    market: Optional[str] = Field(None, description="Market name")
    commodity: str = Field(..., description="Commodity name")
    variety: Optional[str] = Field(None, description="Variety of the commodity")
    grade: Optional[str] = Field(None, description="Grade of the commodity")
    lookback_days: int = Field(90, ge=1, le=365, description="Number of days to look back")
    horizon_days: int = Field(7, ge=1, le=21, description="Number of days to forecast")
    storage_cost_per_quintal_per_day: float = Field(0.8, ge=0, description="Storage cost per quintal per day")
    risk_aversion: float = Field(0.5, ge=0, le=1, description="Risk aversion factor")

class PriceHistoryPoint(BaseModel):
    date: str
    price: float

class PricePrediction(BaseModel):
    nextWeek: float
    nextMonth: float
    recommendation: str

class PriceData(BaseModel):
    crop: str
    currentPrice: float
    previousPrice: float
    unit: str = "quintal"
    marketLocation: str
    lastUpdated: str
    trend: str
    prediction: PricePrediction
    priceHistory: List[PriceHistoryPoint]
    analysis: Optional[Dict[str, Any]] = None

class MandiResponse(BaseModel):
    success: bool
    data: List[PriceData]
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None