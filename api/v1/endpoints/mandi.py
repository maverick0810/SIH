# server/api/v1/endpoints/mandi.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import sys
import os

# Add the server directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.base import agent_registry
from agents.mandi.models import MandiRequest

router = APIRouter()

@router.get("/prices")
async def get_mandi_prices(
    state: str = Query(..., description="State name (e.g., Punjab, Maharashtra)"),
    district: Optional[str] = Query(None, description="District name"),
    market: Optional[str] = Query(None, description="Market name"),
    commodity: str = Query(..., description="Commodity name (e.g., Wheat, Cotton, Rice)"),
    variety: Optional[str] = Query(None, description="Variety of the commodity"),
    grade: Optional[str] = Query(None, description="Grade of the commodity"),
    lookback_days: int = Query(90, ge=1, le=365, description="Number of days to look back for historical data"),
    horizon_days: int = Query(7, ge=1, le=21, description="Number of days to forecast ahead"),
    storage_cost_per_quintal_per_day: float = Query(0.8, ge=0, description="Storage cost per quintal per day"),
    risk_aversion: float = Query(0.5, ge=0, le=1, description="Risk aversion factor (0=risk-seeking, 1=risk-averse)")
):
    """
    Get mandi prices for specified commodity and location
    
    The API accepts any valid state, district, market, and commodity combinations.
    If specific district/market is not provided, the system will use the state-level data.
    """
    try:
        # Get the mandi agent
        mandi_agent = agent_registry.get("mandi")
        if not mandi_agent:
            raise HTTPException(status_code=500, detail="Mandi agent not available")
        
        # Create request with user-provided parameters - no validation or defaults
        request = MandiRequest(
            state=state.strip(),
            district=district.strip() if district else None,
            market=market.strip() if market else None,
            commodity=commodity.strip(),
            variety=variety.strip() if variety else None,
            grade=grade.strip() if grade else None,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            storage_cost_per_quintal_per_day=storage_cost_per_quintal_per_day,
            risk_aversion=risk_aversion
        )
        
        # Execute the request
        response = await mandi_agent.execute(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.get("/commodities")
async def get_commodities(
    state: Optional[str] = Query(None, description="State name to filter commodities")
):
    """
    Get available commodities
    
    Returns a general list of common commodities. The actual availability
    depends on the data.gov.in API and the specific state/market combination.
    """
    # Return common commodities without hardcoded mappings
    commodities = [
        "Wheat", "Rice", "Cotton", "Sugarcane", "Maize", "Soyabean", 
        "Barley", "Mustard", "Groundnut", "Bajra", "Jowar", "Tur", 
        "Gram", "Lentil", "Onion", "Potato", "Tomato", "Brinjal",
        "Garlic", "Ginger", "Turmeric", "Coriander", "Cumin",
        "Chilli", "Coconut", "Arhar", "Moong", "Urad"
    ]
    
    return {
        "success": True,
        "commodities": [{"name": commodity} for commodity in sorted(commodities)],
        "note": "This is a general list. Actual availability depends on the specific state/market."
    }

@router.get("/states")
async def get_states():
    """
    Get available states
    
    Returns common Indian states. The actual data availability
    depends on the data.gov.in API.
    """
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
        "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
        "Uttarakhand", "West Bengal"
    ]
    
    return {
        "success": True,
        "states": [{"name": state} for state in sorted(states)],
        "note": "This is a list of Indian states. Actual data availability depends on the data.gov.in API."
    }

@router.get("/health")
async def mandi_health():
    """Check mandi agent health"""
    try:
        mandi_agent = agent_registry.get("mandi")
        if not mandi_agent:
            return {"status": "unhealthy", "error": "Mandi agent not available"}
        
        health = await mandi_agent.health_check()
        return health
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}