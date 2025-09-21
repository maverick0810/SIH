# Pydantic models for mandi agent
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MandiRequest(BaseModel):
    state: str = "Punjab"
    district: Optional[str] = "Ludhiana"
    market: Optional[str] = "Ludhiana"
    commodity: str = "Wheat"
    variety: Optional[str] = None
    grade: Optional[str] = None
    lookback_days: int = 90
    horizon_days: int = 7
    storage_cost_per_quintal_per_day: float = 0.8
    risk_aversion: float = 0.5

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