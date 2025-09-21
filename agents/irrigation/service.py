# server/agents/irrigation/service.py
"""
Irrigation service - FAO-56 based irrigation planning with weather data
"""
import math
import requests
from typing import List, Dict, Any, Optional
from datetime import date as dt_date, date
from agents.irrigation.models import DailyWeather, IrrigationRequest
import logging

logger = logging.getLogger(__name__)

class IrrigationService:
    """Service for irrigation planning using FAO-56 ET0 calculations and weather data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    # ---------- WEATHER FETCH (Open-Meteo) ----------
    U10_TO_U2 = 0.748  # 10 m → 2 m wind speed conversion
    
    def _safe_num(self, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    
    def _safe_idx(self, seq, i):
        if not isinstance(seq, list):
            return None
        try:
            return seq[i]
        except (IndexError, TypeError):
            return None
    
    def _mul(self, v, k):
        return None if v is None else v * k
    
    def _safe_parse_date(self, s):
        try:
            return dt_date.fromisoformat(s)
        except Exception:
            return None
    
    def fetch_weather(self, request: IrrigationRequest) -> List[DailyWeather]:
        """Fetch daily weather via Open-Meteo"""
        params = {
            "latitude": request.lat,
            "longitude": request.lon,
            "timezone": request.timezone or "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "past_days": 2,
            "forecast_days": max(1, request.planning_horizon_days),
            "daily": ",".join([
                "temperature_2m_min",
                "temperature_2m_max",
                "relative_humidity_2m_mean",
                "wind_speed_10m_mean",
                "precipitation_sum",
                "shortwave_radiation_sum",
            ]),
        }
        
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            logger.info(f"Weather data fetched successfully for lat={request.lat}, lon={request.lon}")
        except requests.RequestException as e:
            logger.error(f"Open-Meteo request failed: {e}")
            raise RuntimeError(f"Weather data fetch failed: {e}") from e
        
        if payload.get("error"):
            error_msg = payload.get("reason", "Open-Meteo error")
            logger.error(f"Open-Meteo API error: {error_msg}")
            raise RuntimeError(error_msg)
        
        d = payload.get("daily", {}) or {}
        times: List[str] = d.get("time", []) or []
        
        daily_weather = []
        for i, t in enumerate(times):
            weather_data = {
                "date": self._safe_parse_date(t),
                "tmin_c": self._safe_num(self._safe_idx(d.get("temperature_2m_min"), i)),
                "tmax_c": self._safe_num(self._safe_idx(d.get("temperature_2m_max"), i)),
                "rh_mean": self._safe_num(self._safe_idx(d.get("relative_humidity_2m_mean"), i)),
                "wind_u2_ms": self._mul(self._safe_num(self._safe_idx(d.get("wind_speed_10m_mean"), i)), self.U10_TO_U2),
                "precip_mm": self._safe_num(self._safe_idx(d.get("precipitation_sum"), i)),
                "rs_mj_m2": self._safe_num(self._safe_idx(d.get("shortwave_radiation_sum"), i)),
            }
            daily_weather.append(DailyWeather(**weather_data))
        
        return daily_weather
    
    # ---------- FAO-56 PM ET0 Calculations ----------
    _GSC = 0.0820          # MJ m-2 min-1
    _SIGMA = 4.903e-9      # MJ K-4 m-2 day-1
    _ALBEDO = 0.23
    
    def _sat_vp_kpa(self, Tc: float) -> float:
        """Saturation vapor pressure"""
        return 0.6108 * math.exp((17.27 * Tc) / (Tc + 237.3))
    
    def _slope_vp_curve_kpa_per_c(self, Tc: float) -> float:
        """Slope of vapor pressure curve"""
        es = self._sat_vp_kpa(Tc)
        return 4098.0 * es / ((Tc + 237.3) ** 2)
    
    def _atm_pressure_kpa(self, z_m: float) -> float:
        """Atmospheric pressure"""
        return 101.3 * (((293.0 - 0.0065 * z_m) / 293.0) ** 5.26)
    
    def _psychrometric_const_kpa_per_c(self, P_kpa: float) -> float:
        """Psychrometric constant"""
        return 0.000665 * P_kpa
    
    def _inv_rel_earth_sun_dist(self, J: int) -> float:
        """Inverse relative distance Earth-Sun"""
        return 1 + 0.033 * math.cos(2 * math.pi * J / 365.0)
    
    def _solar_declination(self, J: int) -> float:
        """Solar declination"""
        return 0.409 * math.sin(2 * math.pi * J / 365.0 - 1.39)
    
    def _sunset_hour_angle(self, phi_rad: float, delta: float) -> float:
        """Sunset hour angle"""
        x = -math.tan(phi_rad) * math.tan(delta)
        x = max(-1.0, min(1.0, x))
        return math.acos(x)
    
    def _extraterrestrial_radiation_MJm2day(self, lat_deg: float, J: int) -> float:
        """Extraterrestrial radiation"""
        phi = math.radians(lat_deg)
        dr = self._inv_rel_earth_sun_dist(J)
        delta = self._solar_declination(J)
        ws = self._sunset_hour_angle(phi, delta)
        Ra = (24 * 60 / math.pi) * self._GSC * dr * (
            ws * math.sin(phi) * math.sin(delta) +
            math.cos(phi) * math.cos(delta) * math.sin(ws)
        )
        return Ra
    
    def _clear_sky_rad_Rso(self, Ra: float, z_m: float) -> float:
        """Clear sky radiation"""
        return (0.75 + 2e-5 * z_m) * Ra
    
    def _net_shortwave_Rns(self, Rs: float) -> float:
        """Net shortwave radiation"""
        return (1 - self._ALBEDO) * Rs
    
    def _net_longwave_Rnl(self, Tmax_c: float, Tmin_c: float, ea_kpa: float, Rs: float, Rso: float) -> float:
        """Net longwave radiation"""
        TmaxK = Tmax_c + 273.16
        TminK = Tmin_c + 273.16
        term_temp = (TmaxK**4 + TminK**4) / 2.0
        term_cloud = 1.35 * min(Rs / max(Rso, 1e-6), 1.0) - 0.35
        term_vp = 0.34 - 0.14 * math.sqrt(max(ea_kpa, 0.0))
        return self._SIGMA * term_temp * term_vp * term_cloud
    
    def compute_et0(self, daily_weather: List[DailyWeather], lat: float, elevation_m: float) -> List[Dict[str, Any]]:
        """Compute FAO-56 Penman-Monteith ET0"""
        et0_rows = []
        z = float(elevation_m or 0.0)
        P = self._atm_pressure_kpa(z)
        gamma = self._psychrometric_const_kpa_per_c(P)
        
        for d in daily_weather:
            if not (d and d.date and d.tmin_c is not None and d.tmax_c is not None
                    and d.rh_mean is not None and d.wind_u2_ms is not None
                    and d.rs_mj_m2 is not None):
                et0_rows.append({"date": d.date.isoformat() if d and d.date else None, "et0": None})
                continue
            
            tmin = float(d.tmin_c)
            tmax = float(d.tmax_c)
            tmean = (tmin + tmax) / 2.0
            rh = max(0.0, min(100.0, float(d.rh_mean)))
            u2 = max(0.0, float(d.wind_u2_ms))
            Rs = max(0.0, float(d.rs_mj_m2))
            
            Delta = self._slope_vp_curve_kpa_per_c(tmean)
            es_tmin = self._sat_vp_kpa(tmin)
            es_tmax = self._sat_vp_kpa(tmax)
            es = (es_tmin + es_tmax) / 2.0
            ea = (rh / 100.0) * self._sat_vp_kpa(tmean)
            
            J = d.date.timetuple().tm_yday
            Ra = self._extraterrestrial_radiation_MJm2day(lat, J)
            Rso = self._clear_sky_rad_Rso(Ra, z)
            Rns = self._net_shortwave_Rns(Rs)
            Rnl = self._net_longwave_Rnl(tmax, tmin, ea, Rs, Rso)
            Rn = max(0.0, Rns - Rnl)
            G = 0.0
            
            num1 = 0.408 * Delta * (Rn - G)
            num2 = gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
            den = Delta + gamma * (1.0 + 0.34 * u2)
            et0 = (num1 + num2) / max(den, 1e-6)
            
            et0_rows.append({
                "date": d.date.isoformat(),
                "et0": round(max(0.0, et0), 2)
            })
        
        return et0_rows
    
    def _kc_for_crop(self, crop: str, days_after_sowing: int) -> float:
        """Crop coefficient based on growth stage"""
        crop = crop.lower()
        
        # Enhanced crop coefficient curves
        crop_kc_curves = {
            "wheat": {
                "initial": (0, 30, 0.4),      # (start_day, end_day, kc)
                "development": (30, 60, 0.7), 
                "mid": (60, 120, 1.15),
                "late": (120, 150, 0.6)
            },
            "rice": {
                "initial": (0, 20, 1.05),
                "development": (20, 35, 1.15),
                "mid": (35, 105, 1.2),
                "late": (105, 135, 0.75)
            },
            "cotton": {
                "initial": (0, 30, 0.35),
                "development": (30, 65, 0.75),
                "mid": (65, 130, 1.15),
                "late": (130, 180, 0.7)
            },
            "maize": {
                "initial": (0, 25, 0.3),
                "development": (25, 50, 0.7),
                "mid": (50, 100, 1.2),
                "late": (100, 130, 0.6)
            },
            "tomato": {
                "initial": (0, 25, 0.6),
                "development": (25, 45, 0.8),
                "mid": (45, 100, 1.15),
                "late": (100, 130, 0.8)
            }
        }
        
        # Default curve for unknown crops
        default_curve = {
            "initial": (0, 30, 0.4),
            "development": (30, 60, 0.8),
            "mid": (60, 120, 1.1),
            "late": (120, 150, 0.7)
        }
        
        curve = crop_kc_curves.get(crop, default_curve)
        
        # Determine growth stage and return Kc
        for stage, (start, end, kc) in curve.items():
            if start <= days_after_sowing < end:
                return kc
        
        # If beyond all stages, use late stage Kc
        return curve["late"][2]
    
    def compute_etc(self, et0_daily: List[Dict[str, Any]], crop: str, sowing_date: dt_date) -> List[Dict[str, Any]]:
        """Compute crop evapotranspiration (ETc)"""
        etc_daily = []
        
        for row in et0_daily:
            et0_val = row.get("et0")
            date_str = row.get("date")
            
            if date_str and et0_val is not None:
                current_date = dt_date.fromisoformat(date_str)
                days_after_sowing = (current_date - sowing_date).days
                kc = self._kc_for_crop(crop, days_after_sowing)
                etc = et0_val * kc
            else:
                kc = None
                etc = None
            
            etc_daily.append({
                "date": date_str,
                "kc": round(kc, 2) if kc is not None else None,
                "etc": round(etc, 2) if etc is not None else None,
            })
        
        return etc_daily
    
    def plan_irrigation(self, daily_weather: List[DailyWeather], etc_daily: List[Dict[str, Any]], 
                       rain_skip_mm: float, target_depth_mm: float) -> Dict[str, Any]:
        """Plan irrigation schedule"""
        irrigation_events = []
        rain_events = []
        running_deficit = 0.0
        
        for i, d in enumerate(daily_weather):
            etc = etc_daily[i].get("etc") or 0.0
            rain = d.precip_mm or 0.0
            
            # Track significant rain events
            if rain >= rain_skip_mm:
                rain_events.append({
                    "date": d.date.isoformat() if d.date else None,
                    "mm": round(rain, 1)
                })
                running_deficit = max(0, running_deficit - rain)  # Rain reduces deficit
            
            # Add daily ET deficit
            running_deficit += etc - min(rain, etc)  # Don't double-count rain reduction
            
            # Schedule irrigation when deficit reaches target
            if running_deficit >= target_depth_mm:
                confidence = "high" if i <= 3 else "medium" if i <= 7 else "low"
                irrigation_events.append({
                    "date": d.date.isoformat() if d.date else None,
                    "mm": target_depth_mm,
                    "reason": f"Deficit reached {target_depth_mm}mm",
                    "confidence": confidence
                })
                running_deficit = 0.0
        
        return {
            "irrigation_events": irrigation_events,
            "rain_events": rain_events,
            "deficit_end_mm": round(running_deficit, 1),
            "total_irrigation_mm": sum(event["mm"] for event in irrigation_events),
            "next_irrigation_date": irrigation_events[0]["date"] if irrigation_events else None
        }