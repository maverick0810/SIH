# server/agents/mandi/service.py
"""
Mandi data service using the existing working LangGraph agent
"""
import sys
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil import tz


# Add server directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import your existing working functions
try:
    # Add the root directory to find your existing file
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.append(root_dir)
    
    from langgraph_mandi_agent_programmatic import (
        MarketAgent as OriginalMarketAgent,
        MarketInput as OriginalMarketInput,
        fetch_agri_data,
        build_series,
        compute_features,
        heuristic_forecast
    )
    
    ORIGINAL_AGENT_AVAILABLE = True
    print("✅ Successfully imported original LangGraph agent")
except ImportError as e:
    print(f"⚠️ Could not import original agent: {e}")
    print("Will use fallback implementation")
    ORIGINAL_AGENT_AVAILABLE = False

from core.exceptions import ExternalAPIError

IST = tz.gettz("Asia/Kolkata")

class MandiDataService:
    """Service that uses your existing working LangGraph agent"""
    
    def __init__(self, api_key: Optional[str], config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config
        
        if ORIGINAL_AGENT_AVAILABLE:
            self.original_agent = OriginalMarketAgent(api_key=api_key)
            print(f"✅ Initialized original agent with API key: {'Yes' if api_key else 'No'}")
        else:
            self.original_agent = None
            print("⚠️ Using fallback agent implementation")
    
    def fetch_paginated_data(self, request) -> List[Dict[str, Any]]:
        """Use your existing agent to fetch data"""
        if not self.original_agent:
            print("🔄 Using fallback data - original agent not available")
            return []  # Fallback to demo data
        
        try:
            print(f"🚀 Using original agent for: {request.state}/{request.commodity}")
            
            # Convert our request to the original format
            original_input = OriginalMarketInput(
                state=request.state,
                district=request.district,
                market=request.market,
                commodity=request.commodity,
                variety=request.variety,
                grade=request.grade,
                lookback_days=request.lookback_days,
                horizon_days=request.horizon_days,
                storage_cost_per_quintal_per_day=request.storage_cost_per_quintal_per_day,
                risk_aversion=request.risk_aversion
            )
            
            # Use your existing agent
            result = self.original_agent.run(original_input)
            print(f"✅ Original agent returned result with confidence: {result.confidence}")
            
            # Return the result wrapped for processing
            return [{"original_result": result}]
            
        except Exception as e:
            print(f"❌ Error using original agent: {e}")
            return []
    
    def build_price_series(self, raw_data, lookback_days):
        """Use original build_series function or fallback"""
        if raw_data and len(raw_data) > 0 and "original_result" in raw_data[0]:
            # Data already processed by original agent
            return pd.DataFrame({"modal": []})  # Return empty to signal processed data
        
        if ORIGINAL_AGENT_AVAILABLE and raw_data:
            try:
                return build_series(raw_data, lookback_days)
            except Exception as e:
                print(f"Error in build_series: {e}")
                return pd.DataFrame({"modal": []})
        
        # Fallback implementation
        return self._fallback_build_series(raw_data, lookback_days)
    
    def _fallback_build_series(self, raw_data, lookback_days):
        """Fallback implementation when original agent not available"""
        if not raw_data:
            return pd.DataFrame({"modal": []})
        
        try:
            df = pd.DataFrame(raw_data)
            
            # Try to find date and price columns
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'arrival' in col.lower()]
            price_cols = [col for col in df.columns if 'price' in col.lower() or 'modal' in col.lower()]
            
            if not date_cols or not price_cols:
                return pd.DataFrame({"modal": []})
            
            # Simple processing
            df_clean = pd.DataFrame({
                "date": pd.to_datetime(df[date_cols[0]], errors='coerce').dt.date,
                "modal": pd.to_numeric(df[price_cols[0]], errors='coerce')
            }).dropna()
            
            if df_clean.empty:
                return pd.DataFrame({"modal": []})
            
            # Filter by lookback period
            today = datetime.now(tz=IST).date()
            start_date = today - timedelta(days=lookback_days)
            df_clean = df_clean[df_clean["date"] >= start_date]
            
            # Set date as index
            df_clean.set_index("date", inplace=True)
            return df_clean
            
        except Exception as e:
            print(f"Error in fallback build_series: {e}")
            return pd.DataFrame({"modal": []})
    
    def compute_features(self, price_series):
        """Use original compute_features function or fallback"""
        if ORIGINAL_AGENT_AVAILABLE:
            try:
                return compute_features(price_series)
            except Exception as e:
                print(f"Error in compute_features: {e}")
                return self._fallback_compute_features(price_series)
        
        return self._fallback_compute_features(price_series)
    
    def _fallback_compute_features(self, price_series):
        """Fallback feature computation"""
        s = price_series.dropna()
        
        if len(s) == 0:
            return {"volatility_30d": None, "trend_delta": None, "num_points": 0, "missing_rate": 1.0}
        
        features = {
            "num_points": len(s),
            "missing_rate": price_series.isna().mean()
        }
        
        try:
            if len(s) >= 7:
                sma7 = s.rolling(7, min_periods=1).mean().iloc[-1]
                sma30 = s.rolling(30, min_periods=1).mean().iloc[-1] if len(s) >= 30 else sma7
                
                if sma30 != 0:
                    features["trend_delta"] = (sma7 - sma30) / sma30
            
            if len(s) >= 14:
                volatility = s.rolling(14, min_periods=1).std().iloc[-1]
                mean_price = s.rolling(30, min_periods=1).mean().iloc[-1]
                
                if mean_price != 0:
                    features["volatility_30d"] = volatility / mean_price
        except Exception as e:
            print(f"Error computing features: {e}")
        
        return features
    
    def heuristic_forecast(self, price_series, horizon_days):
        """Use original heuristic_forecast function or fallback"""
        if ORIGINAL_AGENT_AVAILABLE:
            try:
                return heuristic_forecast(price_series, horizon_days)
            except Exception as e:
                print(f"Error in heuristic_forecast: {e}")
                return self._fallback_forecast(price_series, horizon_days)
        
        return self._fallback_forecast(price_series, horizon_days)
    
    def _fallback_forecast(self, price_series, horizon_days):
        """Fallback forecast implementation"""
        s = price_series.dropna()
        
        if len(s) < 7:
            last_price = s.iloc[-1] if len(s) > 0 else 2450.0
            return {
                "mean": last_price,
                "low": last_price * 0.95,
                "high": last_price * 1.05
            }
        
        try:
            # Simple moving average forecast
            sma7 = s.rolling(7).mean().iloc[-1]
            volatility = s.rolling(14).std().iloc[-1] if len(s) >= 14 else s.std()
            
            forecast_price = sma7
            uncertainty = volatility * 1.5 if pd.notna(volatility) else forecast_price * 0.05
            
            return {
                "mean": forecast_price,
                "low": max(0, forecast_price - uncertainty),
                "high": forecast_price + uncertainty
            }
        except Exception as e:
            print(f"Error in fallback forecast: {e}")
            return {"mean": 2450, "low": 2400, "high": 2500}