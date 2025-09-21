# server/api/v1/endpoints/mandi.py
from fastapi import APIRouter, HTTPException
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
    state: str = "Punjab",
    district: Optional[str] = "Ludhiana",
    market: Optional[str] = "Ludhiana",
    commodity: str = "Wheat"
):
    """Get mandi prices"""
    try:
        mandi_agent = agent_registry.get("mandi")
        if not mandi_agent:
            raise HTTPException(status_code=500, detail="Mandi agent not available")
        
        request = MandiRequest(
            state=state,
            district=district,
            market=market,
            commodity=commodity
        )
        
        response = await mandi_agent.execute(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/commodities")
async def get_commodities():
    """Get available commodities"""
    return {
        "success": True,
        "commodities": [
            {"name": "Wheat", "markets": ["Ludhiana", "Amritsar"]},
            {"name": "Rice", "markets": ["Chandigarh", "Ludhiana"]},
            {"name": "Cotton", "markets": ["Bathinda", "Mansa"]}
        ]
    }