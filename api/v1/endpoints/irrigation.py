# server/api/v1/endpoints/irrigation.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date, datetime, timedelta
import sys
import os

# Add the server directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.base import agent_registry
from agents.irrigation.models import IrrigationRequest

router = APIRouter()

@router.get("/plan")
async def get_irrigation_plan(
    lat: float = Query(..., description="Latitude of the location"),
    lon: float = Query(..., description="Longitude of the location"),
    crop: str = Query(..., description="Crop type (e.g., wheat, rice, cotton, maize, tomato)"),
    sowing_date: date = Query(..., description="Date when crop was sown (YYYY-MM-DD)"),
    timezone: str = Query("auto", description="Timezone for the location"),
    soil_texture: str = Query("loam", description="Soil texture type (sand, sandy_loam, loam, clay_loam, clay)"),
    elevation_m: float = Query(100.0, ge=0, le=5000, description="Elevation in meters"),
    planning_horizon_days: int = Query(10, ge=1, le=21, description="Number of days to plan ahead"),
    rain_skip_mm: float = Query(5.0, ge=0, description="Minimum rain amount to skip irrigation (mm)"),
    target_depth_mm: float = Query(20.0, ge=5, le=80, description="Target irrigation depth (mm)")
):
    """
    Get irrigation plan for specified crop and location
    
    Uses FAO-56 Penman-Monteith method to calculate crop water requirements
    and provides intelligent irrigation scheduling based on weather forecasts.
    """
    try:
        # Get the irrigation agent
        irrigation_agent = agent_registry.get("irrigation")
        if not irrigation_agent:
            raise HTTPException(status_code=500, detail="Irrigation agent not available")
        
        # Validate soil texture
        valid_soils = ["sand", "sandy_loam", "loam", "clay_loam", "clay"]
        if soil_texture not in valid_soils:
            raise HTTPException(status_code=400, detail=f"Invalid soil_texture. Must be one of: {valid_soils}")
        
        # Create request
        request = IrrigationRequest(
            lat=lat,
            lon=lon,
            timezone=timezone,
            crop=crop.strip(),
            sowing_date=sowing_date,
            soil_texture=soil_texture,
            elevation_m=elevation_m,
            planning_horizon_days=planning_horizon_days,
            rain_skip_mm=rain_skip_mm,
            target_depth_mm=target_depth_mm
        )
        
        # Execute the request
        response = await irrigation_agent.execute(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing irrigation request: {str(e)}")

@router.get("/crops")
async def get_crop_recommendations():
    """Get available crop types with their characteristics"""
    try:
        irrigation_agent = agent_registry.get("irrigation")
        if not irrigation_agent:
            raise HTTPException(status_code=500, detail="Irrigation agent not available")
        
        crops = await irrigation_agent.get_crop_recommendations()
        return {
            "success": True,
            "crops": crops,
            "note": "These are supported crops with optimized irrigation parameters"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting crop recommendations: {str(e)}")

@router.get("/soils")
async def get_soil_types():
    """Get available soil texture types"""
    try:
        irrigation_agent = agent_registry.get("irrigation")
        if not irrigation_agent:
            raise HTTPException(status_code=500, detail="Irrigation agent not available")
        
        soils = await irrigation_agent.get_soil_types()
        return {
            "success": True,
            "soil_types": soils,
            "note": "Soil texture affects water retention and irrigation frequency"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting soil types: {str(e)}")

@router.get("/weather-preview")
async def get_weather_preview(
    lat: float = Query(..., description="Latitude of the location"),
    lon: float = Query(..., description="Longitude of the location"),
    days: int = Query(7, ge=1, le=14, description="Number of days to preview")
):
    """Get weather preview for a location"""
    try:
        from agents.irrigation.service import IrrigationService
        from agents.irrigation.models import IrrigationRequest
        from datetime import date
        
        # Create temporary request for weather preview
        temp_request = IrrigationRequest(
            lat=lat,
            lon=lon,
            crop="wheat",  # dummy crop
            sowing_date=date.today(),
            planning_horizon_days=days
        )
        
        service = IrrigationService({})
        weather_data = service.fetch_weather(temp_request)
        
        # Format for response
        weather_preview = []
        for d in weather_data[:days]:
            weather_preview.append({
                "date": d.date.isoformat() if d.date else None,
                "temp_min_c": d.tmin_c,
                "temp_max_c": d.tmax_c,
                "humidity_pct": d.rh_mean,
                "wind_speed_ms": d.wind_u2_ms,
                "precipitation_mm": d.precip_mm,
                "solar_radiation_mj": d.rs_mj_m2
            })
        
        return {
            "success": True,
            "weather": weather_preview,
            "location": f"({lat:.3f}, {lon:.3f})",
            "note": "Weather forecast from Open-Meteo"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting weather preview: {str(e)}")

@router.get("/health")
async def irrigation_health():
    """Check irrigation agent health"""
    try:
        irrigation_agent = agent_registry.get("irrigation")
        if not irrigation_agent:
            return {"status": "unhealthy", "error": "Irrigation agent not available"}
        
        health = await irrigation_agent.health_check()
        return health
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}