# server/agents/mandi/agent.py
"""
Mandi price prediction and analysis agent - CORRECTED VERSION
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import requests

# Add server directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base import BaseAgent
from agents.mandi.models import MandiRequest, MandiResponse, PriceData, PricePrediction, PriceHistoryPoint
from agents.mandi.service import MandiDataService
from core.exceptions import AgentError, AgentConfigError

class MandiAgent(BaseAgent[MandiRequest, MandiResponse]):
    """
    Mandi price prediction agent - FIXED to properly use real API data
    
    Features:
    - Real-time price fetching from data.gov.in via original agent
    - Price trend analysis
    - AI-powered price predictions
    - Buy/Sell/Hold recommendations
    - Historical price tracking
    - Fallback to demo data only when API fails
    """
    
    def __init__(self):
        super().__init__("mandi")
        
        # CRITICAL: Better API key validation
        api_key = self.settings.data_gov_in_api_key
        print(f"🔑 MandiAgent - API Key check: {'Available' if api_key else 'NOT AVAILABLE'}")
        
        if not api_key:
            print("❌ CRITICAL: No DATA_GOV_IN_API_KEY found!")
            print("❌ Agent will only return fallback/demo data")
            print("❌ Please check your .env file")
        else:
            print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-5:]}")
        
        self.data_service = MandiDataService(
            api_key=api_key,
            config=self.config
        )
        self.logger.info("Mandi agent initialized with data service")
    
    def _validate_config(self) -> None:
        """Validate mandi agent configuration"""
        required_config = [
            "default_state", "default_commodity", "lookback_days", 
            "horizon_days", "storage_cost_per_quintal_per_day"
        ]
        
        missing = [key for key in required_config if key not in self.config]
        if missing:
            raise AgentConfigError(f"Missing mandi config: {missing}")
        
        # ENHANCED: Better API key validation
        if not self.settings.data_gov_in_api_key:
            self.logger.warning("❌ CRITICAL: No data.gov.in API key provided")
            self.logger.warning("❌ Agent will only return fallback/demo data")
            self.logger.warning("❌ Set DATA_GOV_IN_API_KEY in your .env file")
        else:
            self.logger.info("✅ data.gov.in API key is available")
    
    async def process_request(self, request: MandiRequest) -> MandiResponse:
        """Process mandi price request using original agent or fallback"""
        
        # ENHANCED: Better logging and error detection
        print(f"🚀 Processing mandi request: {request.state}/{request.commodity}")
        print(f"🔑 API Key available: {'Yes' if self.settings.data_gov_in_api_key else 'No'}")
        
        try:
            # Fetch market data using the service (which uses original agent)
            raw_data = await self._fetch_market_data(request)
            
            # ENHANCED: Better detection of real vs fallback data
            has_real_data = raw_data and len(raw_data) > 0 and "original_result" in raw_data[0]
            
            if has_real_data:
                original_result = raw_data[0]["original_result"]
                num_points = original_result.diagnostics.get("num_points", 0)
                
                if num_points > 0:
                    print(f"✅ Using REAL API data with {num_points} data points")
                    analysis_result = await self._analyze_market_data(raw_data, request)
                    message = f"Real market data retrieved and analyzed successfully ({num_points} data points)"
                    is_real_data = True
                else:
                    print("⚠️ Original agent returned 0 data points - using fallback")
                    analysis_result = self._generate_fallback_price_data(request)
                    message = "No real data available from API - using demo data"
                    is_real_data = False
            else:
                print("⚠️ No data from original agent - using fallback")
                analysis_result = self._generate_fallback_price_data(request)
                message = "API unavailable or failed - using demo data"
                is_real_data = False
            
            # Create response with proper metadata
            response = MandiResponse(
                success=True,
                data=[analysis_result],
                message=message,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "request_params": request.dict(),
                    "data_points": num_points if has_real_data else 0,
                    "analysis_method": "original_agent" if is_real_data else "fallback",
                    "real_data": is_real_data,
                    "api_key_available": bool(self.settings.data_gov_in_api_key)
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing mandi request: {e}")
            print(f"❌ Error in process_request: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            raise AgentError(f"Failed to process mandi request: {e}")
    
    async def _fetch_market_data(self, request: MandiRequest) -> List[Dict[str, Any]]:
        """Fetch raw market data from external API via data service"""
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            self.data_service.fetch_paginated_data,
            request
        )
    
    async def _analyze_market_data(self, raw_data: List[Dict[str, Any]], request: MandiRequest) -> PriceData:
        """Analyze market data and generate predictions"""
        
        # Check if we have data from the original agent
        if raw_data and len(raw_data) > 0 and "original_result" in raw_data[0]:
            self.logger.info("✅ Using original agent result")
            return self._convert_original_result_to_price_data(raw_data[0]["original_result"], request)
        
        # If no real data, use fallback processing
        self.logger.info("⚠️ Using fallback data processing")
        return await self._fallback_analyze_market_data(raw_data, request)
    
    def _convert_original_result_to_price_data(self, original_result, request: MandiRequest) -> PriceData:
        """Convert your original agent result to our PriceData format"""
        
        print(f"🔄 Converting original result with confidence: {original_result.confidence}")
        print(f"📊 Diagnostics: {original_result.diagnostics}")
        
        # Extract data from your original agent's output
        forecast = original_result.forecast
        action = original_result.action
        diagnostics = original_result.diagnostics
        drivers = original_result.drivers
        
        current_price = diagnostics.get("p_now", 2450)
        
        # Calculate previous price (approximate from trend if available)
        trend_delta = diagnostics.get("trend_delta", 0)
        previous_price = current_price / (1 + trend_delta) if trend_delta else current_price * 0.98
        
        # Map action percentages to recommendation
        sell_pct = action.sell_now_pct
        if sell_pct >= 70:
            recommendation = "sell"
        elif sell_pct <= 30:
            recommendation = "buy" 
        else:
            recommendation = "hold"
        
        # Create prediction using forecast data
        prediction = PricePrediction(
            nextWeek=round(forecast.modal, 2),
            nextMonth=round(forecast.high, 2),
            recommendation=recommendation
        )
        
        # Calculate trend from action percentages
        if sell_pct > 60:
            trend = "down"
        elif sell_pct < 40:
            trend = "up"
        else:
            trend = "stable"
        
        # Generate price history based on current price and trend
        price_history = self._generate_realistic_price_history(current_price, trend)
        
        return PriceData(
            crop=request.commodity,
            currentPrice=round(current_price, 2),
            previousPrice=round(previous_price, 2),
            unit="quintal",
            marketLocation=f"{forecast.mandi_used or request.market or request.district} Mandi",
            lastUpdated="Real-time data",
            trend=trend,
            prediction=prediction,
            priceHistory=price_history,
            analysis={
                "confidence": round(original_result.confidence, 2),
                "volatility": diagnostics.get("volatility_30d"),
                "trend_strength": diagnostics.get("trend_delta"),
                "data_quality": {
                    "points": diagnostics.get("num_points", 0),
                    "missing_rate": diagnostics.get("missing_rate", 0.0)
                },
                "drivers": drivers,
                "original_forecast": {
                    "modal": forecast.modal,
                    "low": forecast.low,
                    "high": forecast.high,
                    "horizon_days": forecast.horizon_days
                },
                "original_action": {
                    "sell_now_pct": action.sell_now_pct,
                    "hold_pct": action.hold_pct,
                    "window": action.window
                },
                "data_source": "real_api"
            }
        )
    
    async def _fallback_analyze_market_data(self, raw_data: List[Dict[str, Any]], request: MandiRequest) -> PriceData:
        """Fallback analysis when original agent data not available"""
        
        if not raw_data:
            return self._generate_fallback_price_data(request)
        
        # Build time series
        df = self._build_price_series(raw_data, request.lookback_days)
        
        if df.empty or df['modal'].isna().all():
            return self._generate_fallback_price_data(request)
        
        # Extract current price
        current_price = float(df['modal'].dropna().iloc[-1])
        previous_price = float(df['modal'].dropna().iloc[-2]) if len(df['modal'].dropna()) > 1 else current_price * 0.98
        
        # Generate features and forecast
        features = self._compute_features(df['modal'])
        forecast = self._heuristic_forecast(df['modal'], request.horizon_days)
        
        # Make trading decision
        decision = self._decide_action(
            current_price=current_price,
            predicted_price=forecast["mean"],
            volatility=features.get("volatility_30d"),
            horizon_days=request.horizon_days,
            storage_cost=request.storage_cost_per_quintal_per_day,
            risk_aversion=request.risk_aversion
        )
        
        # Calculate trend
        trend = self._calculate_trend(current_price, previous_price)
        
        # Generate price history
        price_history = self._generate_price_history(df['modal'])
        
        # Create prediction
        prediction = PricePrediction(
            nextWeek=round(forecast["mean"], 2),
            nextMonth=round(forecast["high"], 2),
            recommendation=decision["recommendation"]
        )
        
        # Create price data
        price_data = PriceData(
            crop=request.commodity,
            currentPrice=round(current_price, 2),
            previousPrice=round(previous_price, 2),
            unit="quintal",
            marketLocation=f"{request.market or request.district} Mandi",
            lastUpdated="Processed data",
            trend=trend,
            prediction=prediction,
            priceHistory=price_history,
            analysis={
                "confidence": self._calculate_confidence(features),
                "volatility": features.get("volatility_30d"),
                "trend_strength": features.get("trend_delta"),
                "data_quality": {
                    "points": features.get("num_points", 0),
                    "missing_rate": features.get("missing_rate", 1.0)
                },
                "note": "Processed using fallback analysis",
                "data_source": "fallback_processing"
            }
        )
        
        return price_data
    
    def _generate_realistic_price_history(self, current_price: float, trend: str) -> List[PriceHistoryPoint]:
        """Generate realistic price history based on current price and trend"""
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        history = []
        
        # Generate prices with trend
        if trend == "up":
            base_prices = [0.94, 0.96, 0.98, 1.01, 1.00]  # Upward trend
        elif trend == "down":
            base_prices = [1.06, 1.04, 1.02, 0.99, 1.00]  # Downward trend
        else:
            base_prices = [0.98, 0.99, 1.01, 1.00, 1.00]  # Stable
        
        for i, day in enumerate(days):
            price = current_price * base_prices[i]
            # Add small random variation (±1%)
            variation = 1 + (0.02 * (i % 3 - 1) / 2)
            price *= variation
            history.append(PriceHistoryPoint(date=day, price=round(price, 2)))
        
        return history
    
    def _generate_fallback_price_data(self, request: MandiRequest) -> PriceData:
        """Generate fallback price data when no real data is available"""
        # Base prices by commodity
        base_prices = {
            "Wheat": 2450.0,
            "Rice": 3200.0,
            "Cotton": 5800.0,
            "Sugarcane": 380.0,
            "Maize": 1850.0,
            "Soyabean": 4500.0
        }
        
        base_price = base_prices.get(request.commodity, 2450.0)
        
        prediction = PricePrediction(
            nextWeek=base_price * 1.02,
            nextMonth=base_price * 1.05,
            recommendation="hold"
        )
        
        price_history = [
            PriceHistoryPoint(date="Mon", price=base_price * 0.96),
            PriceHistoryPoint(date="Tue", price=base_price * 0.98),
            PriceHistoryPoint(date="Wed", price=base_price * 0.99),
            PriceHistoryPoint(date="Thu", price=base_price * 1.01),
            PriceHistoryPoint(date="Fri", price=base_price)
        ]
        
        return PriceData(
            crop=request.commodity,
            currentPrice=base_price,
            previousPrice=base_price * 0.98,
            unit="quintal",
            marketLocation=f"{request.market or request.district} Mandi (Demo Data)",
            lastUpdated="Demo data",
            trend="stable",
            prediction=prediction,
            priceHistory=price_history,
            analysis={
                "confidence": 0.3,
                "note": "Demo data - API key not available or API failed",
                "data_source": "demo_fallback"
            }
        )
    
    def _build_price_series(self, raw_data: List[Dict[str, Any]], lookback_days: int) -> pd.DataFrame:
        """Build time series from raw data"""
        return self.data_service.build_price_series(raw_data, lookback_days)
    
    def _compute_features(self, price_series: pd.Series) -> Dict[str, Any]:
        """Compute technical features from price series"""
        return self.data_service.compute_features(price_series)
    
    def _heuristic_forecast(self, price_series: pd.Series, horizon_days: int) -> Dict[str, float]:
        """Generate price forecast"""
        return self.data_service.heuristic_forecast(price_series, horizon_days)
    
    def _decide_action(
        self, 
        current_price: float, 
        predicted_price: float, 
        volatility: Optional[float],
        horizon_days: int,
        storage_cost: float,
        risk_aversion: float
    ) -> Dict[str, Any]:
        """Make buy/sell/hold decision"""
        
        # Calculate expected profit
        expected_profit = (predicted_price - current_price) - (storage_cost * horizon_days)
        
        # Adjust for risk and volatility
        risk_adjustment = risk_aversion * (volatility or 0.2) * current_price
        adjusted_profit = expected_profit - risk_adjustment
        
        # Decision threshold (1% of current price)
        threshold = 0.01 * current_price
        
        if adjusted_profit >= threshold:
            recommendation = "buy"
            confidence = min(0.9, 0.5 + (adjusted_profit / current_price))
        elif adjusted_profit <= -threshold:
            recommendation = "sell"
            confidence = min(0.9, 0.5 + abs(adjusted_profit) / current_price)
        else:
            recommendation = "hold"
            confidence = 0.6
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "expected_profit": expected_profit,
            "adjusted_profit": adjusted_profit
        }
    
    def _calculate_trend(self, current: float, previous: float) -> str:
        """Calculate price trend"""
        if current > previous * 1.02:  # 2% increase
            return "up"
        elif current < previous * 0.98:  # 2% decrease
            return "down"
        return "stable"
    
    def _generate_price_history(self, price_series: pd.Series) -> List[PriceHistoryPoint]:
        """Generate price history for charts"""
        if price_series.empty:
            # Generate demo history
            base_price = 2450.0
            return [
                PriceHistoryPoint(date="Mon", price=base_price * 0.96),
                PriceHistoryPoint(date="Tue", price=base_price * 0.98),
                PriceHistoryPoint(date="Wed", price=base_price * 0.99),
                PriceHistoryPoint(date="Thu", price=base_price * 1.01),
                PriceHistoryPoint(date="Fri", price=base_price)
            ]
        
        recent_data = price_series.dropna().tail(7)  # Last 7 days
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        history = []
        for i, (date_idx, price) in enumerate(recent_data.items()):
            day_name = days[i % 7]
            history.append(PriceHistoryPoint(date=day_name, price=round(float(price), 2)))
        
        return history
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate prediction confidence"""
        confidence = 0.5
        
        # Adjust based on data quality
        num_points = features.get("num_points", 0)
        if num_points >= 90:
            confidence += 0.2
        elif num_points >= 30:
            confidence += 0.1
        
        # Adjust based on missing data
        missing_rate = features.get("missing_rate", 1.0)
        if missing_rate < 0.1:
            confidence += 0.1
        elif missing_rate > 0.3:
            confidence -= 0.2
        
        # Adjust based on volatility
        volatility = features.get("volatility_30d")
        if volatility and volatility < 0.15:
            confidence += 0.1
        elif volatility and volatility > 0.3:
            confidence -= 0.1
        
        return max(0.1, min(0.9, confidence))
    
    def get_fallback_response(self, request: MandiRequest, error: Exception) -> MandiResponse:
        """Get fallback response when agent fails"""
        
        fallback_data = self._generate_fallback_price_data(request)
        
        return MandiResponse(
            success=False,
            data=[fallback_data],
            message=f"Using fallback data due to error: {str(error)}",
            timestamp=datetime.now().isoformat(),
            metadata={"fallback": True, "error": str(error)}
        )
    
    async def get_available_commodities(self, state: str = "Punjab") -> List[Dict[str, Any]]:
        """Get available commodities for a state"""
        commodities = [
            {"name": "Wheat", "markets": ["Ludhiana", "Amritsar", "Patiala"]},
            {"name": "Rice", "markets": ["Chandigarh", "Ludhiana", "Amritsar"]},
            {"name": "Cotton", "markets": ["Bathinda", "Mansa", "Faridkot"]},
            {"name": "Sugarcane", "markets": ["Jalandhar", "Kapurthala"]},
            {"name": "Maize", "markets": ["Ludhiana", "Amritsar"]},
            {"name": "Soyabean", "markets": ["Indore", "Dewas", "Ujjain"]}
        ]
        return commodities
    
    async def get_available_markets(self, state: str = "Punjab", commodity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available markets for a state/commodity"""
        markets = [
            {"name": "Ludhiana", "district": "Ludhiana", "commodities": ["Wheat", "Rice", "Maize"]},
            {"name": "Amritsar", "district": "Amritsar", "commodities": ["Wheat", "Rice", "Maize"]},
            {"name": "Chandigarh", "district": "Chandigarh", "commodities": ["Rice", "Wheat"]},
            {"name": "Bathinda", "district": "Bathinda", "commodities": ["Cotton", "Wheat"]},
            {"name": "Patiala", "district": "Patiala", "commodities": ["Wheat", "Rice"]}
        ]
        
        if commodity:
            markets = [m for m in markets if commodity in m["commodities"]]
        
        return markets