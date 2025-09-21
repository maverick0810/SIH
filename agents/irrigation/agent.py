# server/agents/irrigation/agent.py
"""
Irrigation planning agent - FAO-56 based irrigation scheduling
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, date
import logging

# Add server directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base import BaseAgent
from agents.irrigation.models import (
    IrrigationRequest, IrrigationResponse, IrrigationPlan, 
    IrrigationEvent, RainEvent, ET0Data, ETCData
)
from agents.irrigation.service import IrrigationService
from core.exceptions import AgentError, AgentConfigError

class IrrigationAgent(BaseAgent[IrrigationRequest, IrrigationResponse]):
    """
    Irrigation planning agent using FAO-56 methodology
    
    Features:
    - Weather data from Open-Meteo API
    - FAO-56 Penman-Monteith ET0 calculations
    - Crop coefficient (Kc) based on growth stages
    - Soil water deficit tracking
    - Intelligent irrigation scheduling
    - Rain event consideration
    """
    
    def __init__(self):
        super().__init__("irrigation")
        self.service = IrrigationService(config=self.config)
        self.logger.info("Irrigation agent initialized")
    
    def _validate_config(self) -> None:
        """Validate irrigation agent configuration"""
        required_config = [
            "default_planning_horizon_days", "default_rain_skip_mm", 
            "default_target_depth_mm", "default_elevation_m"
        ]
        
        missing = [key for key in required_config if key not in self.config]
        if missing:
            self.logger.warning(f"Missing irrigation config (using defaults): {missing}")
    
    async def process_request(self, request: IrrigationRequest) -> IrrigationResponse:
        """Process irrigation planning request"""
        
        self.logger.info(f"Processing irrigation request for {request.crop} at ({request.lat}, {request.lon})")
        
        try:
            # Step 1: Fetch weather data
            daily_weather = await self._fetch_weather_data(request)
            
            # Step 2: Compute ET0 (reference evapotranspiration)
            et0_daily = self.service.compute_et0(daily_weather, request.lat, request.elevation_m)
            
            # Step 3: Compute ETc (crop evapotranspiration)
            etc_daily = self.service.compute_etc(et0_daily, request.crop, request.sowing_date)
            
            # Step 4: Plan irrigation schedule
            irrigation_plan = self.service.plan_irrigation(
                daily_weather, etc_daily, request.rain_skip_mm, request.target_depth_mm
            )
            
            # Step 5: Create response
            plan_data = IrrigationPlan(
                irrigation_events=[
                    IrrigationEvent(**event) for event in irrigation_plan["irrigation_events"]
                ],
                rain_events=[
                    RainEvent(**event) for event in irrigation_plan["rain_events"]
                ],
                et0_daily=[
                    ET0Data(**row) for row in et0_daily
                ],
                etc_daily=[
                    ETCData(**row) for row in etc_daily
                ],
                deficit_end_mm=irrigation_plan["deficit_end_mm"],
                total_irrigation_mm=irrigation_plan["total_irrigation_mm"],
                next_irrigation_date=irrigation_plan["next_irrigation_date"]
            )
            
            # Determine message based on irrigation needs
            num_irrigations = len(irrigation_plan["irrigation_events"])
            if num_irrigations == 0:
                message = f"No irrigation needed for next {request.planning_horizon_days} days"
            else:
                next_date = irrigation_plan["next_irrigation_date"]
                message = f"{num_irrigations} irrigation(s) recommended. Next: {next_date}"
            
            response = IrrigationResponse(
                success=True,
                data=plan_data,
                message=message,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "request_params": request.dict(),
                    "weather_days": len(daily_weather),
                    "planning_method": "fao56_pm",
                    "crop_stage": self._get_crop_stage(request),
                    "location": f"({request.lat:.3f}, {request.lon:.3f})"
                }
            )
            
            self.logger.info(f"Irrigation plan generated: {num_irrigations} events, deficit: {irrigation_plan['deficit_end_mm']}mm")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing irrigation request: {e}")
            raise AgentError(f"Failed to process irrigation request: {e}")
    
    async def _fetch_weather_data(self, request: IrrigationRequest):
        """Fetch weather data asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            self.service.fetch_weather,
            request
        )
    
    def _get_crop_stage(self, request: IrrigationRequest) -> str:
        """Determine current crop growth stage"""
        days_since_sowing = (date.today() - request.sowing_date).days
        
        # General growth stages (can be refined per crop)
        if days_since_sowing < 30:
            return "initial"
        elif days_since_sowing < 60:
            return "development"
        elif days_since_sowing < 120:
            return "mid-season"
        else:
            return "late-season"
    
    def get_fallback_response(self, request: IrrigationRequest, error: Exception) -> IrrigationResponse:
        """Get fallback response when agent fails"""
        
        # Generate basic fallback irrigation plan
        fallback_plan = IrrigationPlan(
            irrigation_events=[
                IrrigationEvent(
                    date=date.today().isoformat(),
                    mm=request.target_depth_mm,
                    reason="Fallback recommendation",
                    confidence="low"
                )
            ],
            rain_events=[],
            et0_daily=[],
            etc_daily=[],
            deficit_end_mm=request.target_depth_mm,
            total_irrigation_mm=request.target_depth_mm,
            next_irrigation_date=date.today().isoformat()
        )
        
        return IrrigationResponse(
            success=False,
            data=fallback_plan,
            message=f"Using fallback irrigation plan due to error: {str(error)}",
            timestamp=datetime.now().isoformat(),
            metadata={"fallback": True, "error": str(error)}
        )
    
    async def get_crop_recommendations(self) -> List[Dict[str, Any]]:
        """Get available crop types with their characteristics"""
        crops = [
            {
                "name": "Wheat",
                "growth_period_days": 150,
                "water_requirement": "medium",
                "seasons": ["winter", "spring"]
            },
            {
                "name": "Rice", 
                "growth_period_days": 135,
                "water_requirement": "high",
                "seasons": ["summer", "monsoon"]
            },
            {
                "name": "Cotton",
                "growth_period_days": 180,
                "water_requirement": "medium-high",
                "seasons": ["summer"]
            },
            {
                "name": "Maize",
                "growth_period_days": 130,
                "water_requirement": "medium",
                "seasons": ["summer", "winter"]
            },
            {
                "name": "Tomato",
                "growth_period_days": 130,
                "water_requirement": "medium-high",
                "seasons": ["winter", "spring"]
            },
            {
                "name": "Soybean",
                "growth_period_days": 120,
                "water_requirement": "medium",
                "seasons": ["summer"]
            }
        ]
        return crops
    
    async def get_soil_types(self) -> List[Dict[str, Any]]:
        """Get available soil texture types"""
        soils = [
            {
                "type": "sand",
                "description": "Drains quickly, needs frequent irrigation",
                "water_holding_capacity": "low"
            },
            {
                "type": "sandy_loam",
                "description": "Good drainage with moderate water retention",
                "water_holding_capacity": "medium-low"
            },
            {
                "type": "loam",
                "description": "Ideal soil with balanced drainage and retention",
                "water_holding_capacity": "medium"
            },
            {
                "type": "clay_loam",
                "description": "Good water retention, slower drainage",
                "water_holding_capacity": "medium-high"
            },
            {
                "type": "clay",
                "description": "High water retention, poor drainage",
                "water_holding_capacity": "high"
            }
        ]
        return soils