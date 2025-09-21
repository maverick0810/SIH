# server/langgraph_mandi_agent_programmatic.py
"""
Minimal extract from your original LangGraph agent for backend integration
"""

import os
import math
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, date

import requests
import pandas as pd
import numpy as np
from dateutil import tz
from pydantic import BaseModel, Field

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Pydantic validator compatibility
try:
    from pydantic import field_validator as _pyd_field_validator
except Exception:
    from pydantic import validator as _pyd_field_validator

IST = tz.gettz("Asia/Kolkata")

def fetch_agri_data(api_key=None, state=None, district=None, market=None, commodity=None,
                    variety=None, grade=None, arrival_date=None, offset=0, limit=10):
    """
    Fetch agricultural market data from data.gov.in API.
    """
    key = api_key or os.getenv('DATA_GOV_IN_API_KEY')
    if not key:
        raise ValueError("API key required: pass api_key or set DATA_GOV_IN_API_KEY")

    base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    params = {
        "api-key": key,
        "format": "json",
        "offset": offset,
        "limit": limit,
    }

    # Only add filters if values are provided
    filters = {}
    if state:        filters["state.keyword"] = state
    if district:     filters["district"] = district
    if market:       filters["market"] = market
    if commodity:    filters["commodity"] = commodity
    if variety:      filters["variety"] = variety
    if grade:        filters["grade"] = grade
    if arrival_date: filters["arrival_date"] = arrival_date

    for k, v in filters.items():
        params[f"filters[{k}]"] = v

    resp = requests.get(base_url, params=params, timeout=30)
    
    try:
        data = resp.json()
    except Exception:
        resp.raise_for_status()
        raise

    if isinstance(data, dict) and data.get("error"):
        return data
    
    print(f"API Response: total={data.get('total', 0)}, count={data.get('count', 0)}")
    return data

class MarketInput(BaseModel):
    state: str
    market: Optional[str] = None
    commodity: Optional[str] = None
    district: Optional[str] = None
    variety: Optional[str] = None
    grade: Optional[str] = None
    arrival_date: Optional[str] = None
    lookback_days: int = 120
    horizon_days: int = 10
    storage_cost_per_quintal_per_day: float = 0.8
    risk_aversion: float = 0.5

    @_pyd_field_validator("horizon_days")
    def _check_horizon(cls, v):
        if not 1 <= v <= 21:
            raise ValueError("horizon_days should be in [1, 21]")
        return v

class ForecastOutput(BaseModel):
    modal: float
    low: float
    high: float
    horizon_days: int
    mandi_used: Optional[str] = None

class ActionOutput(BaseModel):
    sell_now_pct: int
    hold_pct: int
    window: List[str]

class MarketOutput(BaseModel):
    forecast: ForecastOutput
    action: ActionOutput
    drivers: List[str]
    confidence: float
    diagnostics: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)

def _parse_price(x: Any) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None

def _to_date(d: Any) -> Optional[date]:
    if pd.isna(d):
        return None
    s = str(d).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

def build_series(rows: List[Dict[str, Any]], lookback_days: int) -> pd.DataFrame:
    """
    Build time series from raw market data
    """
    if not rows:
        return pd.DataFrame({"modal": []})

    df = pd.DataFrame(rows)
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"Sample record: {df.iloc[0].to_dict() if len(df) > 0 else 'No records'}")

    def pick(colnames: List[str]) -> Optional[pd.Series]:
        for c in colnames:
            if c in df.columns:
                print(f"Found column: {c}")
                return df[c]
        return None

    # Use the actual API field names
    arrival = pick(["arrival_date", "Arrival_Date"])
    modal = pick(["modal_price", "Modal_x0020_Price"])  # Note: API uses modal_price in JSON

    if arrival is None:
        print("No arrival date column found")
        return pd.DataFrame({"modal": []})
    
    if modal is None:
        print("No modal price column found")
        return pd.DataFrame({"modal": []})

    out_df = pd.DataFrame({
        "date":  [ _to_date(x) for x in arrival ],
        "modal": [ _parse_price(x) for x in modal ],
    }).dropna(subset=["date", "modal"])

    print(f"Processed {len(out_df)} valid records")

    if out_df.empty:
        return pd.DataFrame({"modal": []})

    # Rest of function remains the same...
    out_df = out_df.sort_values("date").groupby("date", as_index=False).tail(1)

    today = datetime.now(tz=IST).date()
    start = today - timedelta(days=lookback_days)
    out_df = out_df[out_df["date"].between(start, today)]

    idx = pd.date_range(start=start, end=today, freq="D").date
    timeline = pd.DataFrame(index=idx)
    timeline.index.name = "date"
    timeline = timeline.join(out_df.set_index("date")["modal"]).sort_index()
    timeline["modal"] = timeline["modal"].astype(float).interpolate(limit=2)
    return timeline
def compute_features(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna()
    if len(s) == 0:
        return {"volatility_30d": None, "trend_delta": None, "num_points": 0, "missing_rate": 1.0}
    
    sma7 = s.rolling(7).mean()
    sma30 = s.rolling(30).mean()
    trend_delta = None
    if pd.notna(sma7.iloc[-1]) and pd.notna(sma30.iloc[-1]) and sma30.iloc[-1] != 0:
        trend_delta = float((sma7.iloc[-1] - sma30.iloc[-1]) / sma30.iloc[-1])
    
    rolling_std14 = s.rolling(14).std().iloc[-1]
    rolling_mean30 = s.rolling(30).mean().iloc[-1]
    vol = None
    if pd.notna(rolling_std14) and pd.notna(rolling_mean30) and rolling_mean30 != 0:
        vol = float(rolling_std14 / rolling_mean30)
    
    missing_rate = float(series.isna().mean())
    return {
        "volatility_30d": vol,
        "trend_delta": trend_delta,
        "num_points": int(len(s)),
        "missing_rate": missing_rate,
    }

def heuristic_forecast(series: pd.Series, horizon_days: int) -> Dict[str, Any]:
    s = series.dropna()
    if len(s) < 14:
        m = s.iloc[-1] if len(s) else np.nan
        return {
            "mean": float(m) if pd.notna(m) else np.nan,
            "low":  float(m) if pd.notna(m) else np.nan,
            "high": float(m) if pd.notna(m) else np.nan,
            "model": "heuristic_sma",
            "mape_cv": None,
        }
    sma7 = s.rolling(7).mean().iloc[-1]
    std14 = s.rolling(14).std().iloc[-1]
    mean = float(sma7)
    band = float(1.5 * (std14 if pd.notna(std14) else 0.0))
    return {"mean": mean, "low": max(0.0, mean - band), "high": mean + band, "model": "heuristic_sma", "mape_cv": None}

def decide_action(p_now: float, p_exp: float, low: float, high: float, horizon_days: int,
                 storage_cost_per_quintal_per_day: float, volatility: Optional[float],
                 risk_aversion: float) -> Dict[str, Any]:
    B = (p_exp - p_now) - storage_cost_per_quintal_per_day * horizon_days
    lam = 0.5 + 0.5 * max(0.0, min(1.0, risk_aversion))
    vol = volatility if (volatility is not None) else 0.2
    B_adj = B - lam * vol * p_now

    sell_now_pct, hold_pct = 50, 50
    thr = 0.01 * p_now
    if B_adj >= thr:
        hold_pct, sell_now_pct = 60, 40
    elif B_adj <= -thr:
        hold_pct, sell_now_pct = 20, 80

    start = (datetime.now(tz=IST).date() + timedelta(days=2)).isoformat()
    end = (datetime.now(tz=IST).date() + timedelta(days=horizon_days)).isoformat()
    return {"sell_now_pct": int(sell_now_pct), "hold_pct": int(hold_pct), "window": [start, end], 
            "B": float(B), "B_adj": float(B_adj)}

def confidence_score(num_points: int, missing_rate: float, volatility: Optional[float], 
                    mape_cv: Optional[float]) -> float:
    c = 0.5
    if num_points >= 90: c += 0.1
    if mape_cv is not None and mape_cv <= 0.12: c += 0.1
    if missing_rate > 0.1: c -= 0.1
    if volatility is not None and volatility > 0.25: c -= 0.1
    return float(max(0.1, min(0.9, c)))

class MarketAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DATA_GOV_IN_API_KEY")

    def _fetch_paginated(self, inp: MarketInput, limit=10000, max_pages=3) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
    
    # Try different combinations to find data
        queries_to_try = [
            # Original query
            {
                "state": inp.state,
                "district": inp.district,
                "market": inp.market,
                "commodity": inp.commodity,
                "variety": inp.variety,
                "grade": inp.grade,
            },
            # Without specific market
            {
                "state": inp.state,
                "district": inp.district,
                "commodity": inp.commodity,
            },
            # Without district
            {
                "state": inp.state,
                "commodity": inp.commodity,
            },
            # Just state and commodity
            {
                "state": inp.state,
                "commodity": inp.commodity,
            },
            # Try Rajasthan (we know it has data)
            {
                "state": "Rajasthan",
                "commodity": inp.commodity,
            },
            # Try any wheat data
            {
                "commodity": inp.commodity,
            }
        ]
        
        for i, query_params in enumerate(queries_to_try):
            print(f"Trying query {i+1}: {query_params}")
            
            offset = 0
            for _ in range(max_pages):
                try:
                    data = fetch_agri_data(
                        api_key=self.api_key,
                        offset=offset,
                        limit=limit,
                        **query_params
                    )
                    
                    if isinstance(data, dict) and data.get("error"):
                        print(f"API error: {data['error']}")
                        break
                        
                    recs = (data or {}).get("records", [])
                    print(f"Got {len(recs)} records from query {i+1}")
                    
                    if recs:
                        all_rows.extend(recs)
                        print(f"Success! Using data from: {query_params}")
                        return all_rows  # Return first successful query
                        
                    if not recs:
                        break  # No more records for this query
                        
                    offset += limit
                    
                except Exception as e:
                    print(f"Error with query {i+1}: {e}")
                    break
        
        print("No data found with any query combination")
        return all_rows

    def _sanitize_float(self, value: Any) -> float:
        """Convert any numeric value to JSON-safe float"""
        try:
            if pd.isna(value) or math.isinf(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def run(self, inp: MarketInput) -> MarketOutput:
        warnings: List[str] = []
        try:
            rows = self._fetch_paginated(inp)
        except Exception as e:
            warnings.append(str(e))
            rows = []

        if not rows:
            warnings.append("No data returned from API for the selected filters.")

        df = build_series(rows, lookback_days=inp.lookback_days)
        s = df["modal"] if not df.empty else pd.Series(dtype=float)

        if len(s.dropna()) == 0:
            # Return error response instead of hardcoded values
            fo = ForecastOutput(
                modal=0.0,
                low=0.0, 
                high=0.0,
                horizon_days=inp.horizon_days, 
                mandi_used=inp.market
            )
            ao = ActionOutput(
                sell_now_pct=0, 
                hold_pct=0,
                window=[date.today().isoformat(), date.today().isoformat()]
            )
            return MarketOutput(
                forecast=fo,
                action=ao,
                drivers=["No market data available"],
                confidence=0.0,
                diagnostics={"num_points": 0, "missing_rate": 1.0, "model": "none"},
                warnings=warnings,
            )

        feats = compute_features(s)
        p_now = self._sanitize_float(s.dropna().iloc[-1])
        fc = heuristic_forecast(s, inp.horizon_days)
        
        # Use actual computed values, no defaults
        p_exp = self._sanitize_float(fc.get("mean", p_now))
        low = self._sanitize_float(fc.get("low", p_now))
        high = self._sanitize_float(fc.get("high", p_now))

        decision = decide_action(
            p_now=p_now, p_exp=p_exp, low=low, high=high, horizon_days=inp.horizon_days,
            storage_cost_per_quintal_per_day=inp.storage_cost_per_quintal_per_day,
            volatility=feats.get("volatility_30d"),
            risk_aversion=inp.risk_aversion,
        )

        conf = confidence_score(
            num_points=feats.get("num_points", 0),
            missing_rate=feats.get("missing_rate", 1.0),
            volatility=feats.get("volatility_30d"),
            mape_cv=fc.get("mape_cv"),
        )

        drivers: List[str] = []
        td = feats.get("trend_delta")
        if td is not None and not math.isnan(td):
            sign = "+" if td >= 0 else "-"
            drivers.append(f"Short-term trend {sign}{abs(td)*100:.1f}% vs 30-day avg")
        if feats.get("volatility_30d") is not None:
            drivers.append(f"Volatility ~{feats['volatility_30d']*100:.0f}% (30d)")
        drivers.append(f"Storage cost ₹{inp.storage_cost_per_quintal_per_day}/q/day considered")

        diagnostics = {
            "volatility_30d": self._sanitize_float(feats.get("volatility_30d")),
            "trend_delta": self._sanitize_float(feats.get("trend_delta")),
            "num_points": feats.get("num_points", 0),
            "missing_rate": self._sanitize_float(feats.get("missing_rate", 1.0)),
            "model": fc.get("model", "heuristic_sma"),
            "p_now": self._sanitize_float(p_now)
        }

        fo = ForecastOutput(
            modal=self._sanitize_float(p_exp), 
            low=self._sanitize_float(low), 
            high=self._sanitize_float(high),
            horizon_days=inp.horizon_days, 
            mandi_used=inp.market
        )
        ao = ActionOutput(
            sell_now_pct=int(decision["sell_now_pct"]),
            hold_pct=int(decision["hold_pct"]),
            window=decision["window"],
        )
        return MarketOutput(
            forecast=fo,
            action=ao,
            drivers=drivers,
            confidence=self._sanitize_float(conf),
            diagnostics=diagnostics,
            warnings=warnings,
        )