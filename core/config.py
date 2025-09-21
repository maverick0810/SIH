# server/core/config.py
"""
Configuration management for backend services - FIXED VERSION
"""

from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
import os
from functools import lru_cache
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # API Configuration
    api_title: str = "KisanSathi AI Backend"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:4173"
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # External API Keys - FIXED: Proper field names and explicit environment reading
    data_gov_in_api_key: Optional[str] = None
    openweather_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # Cache Configuration
    cache_enabled: bool = True
    cache_default_ttl: int = 900  # 15 minutes
    redis_url: Optional[str] = None
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Agent Configurations
    mandi_config: Dict[str, Any] = {
        "default_state": "Punjab",
        "default_district": "Ludhiana",
        "default_market": "Ludhiana",
        "default_commodity": "Wheat",
        "lookback_days": 90,
        "horizon_days": 7,
        "storage_cost_per_quintal_per_day": 0.8,
        "risk_aversion_default": 0.5,
        "max_api_pages": 3,
        "records_per_page": 10000
    }
    
    weather_config: Dict[str, Any] = {
        "default_location": "Ludhiana, Punjab",
        "forecast_days": 7,
        "cache_hours": 1
    }
    
    # Database (if needed later)
    database_url: Optional[str] = None
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # CRITICAL FIX: Manually read environment variables if not set
        if not self.data_gov_in_api_key:
            env_key = os.getenv('DATA_GOV_IN_API_KEY')
            if env_key:
                self.data_gov_in_api_key = env_key
                print(f"✅ Loaded DATA_GOV_IN_API_KEY from environment: {env_key[:10]}...")
            else:
                print("❌ DATA_GOV_IN_API_KEY not found in environment variables")
        
        if not self.openweather_api_key:
            self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for specific agent"""
        config_map = {
            "mandi": self.mandi_config,
            "weather": self.weather_config
        }
        return config_map.get(agent_name, {})
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Validation functions
def validate_api_keys(settings: Settings) -> None:
    """Validate required API keys based on environment"""
    required_keys = []
    
    print(f"🔍 Validating API keys...")
    print(f"🔑 DATA_GOV_IN_API_KEY: {'Set' if settings.data_gov_in_api_key else 'NOT SET'}")
    
    if not settings.data_gov_in_api_key:
        required_keys.append("DATA_GOV_IN_API_KEY")
    
    if required_keys and settings.is_production:
        raise ValueError(f"Missing required API keys in production: {', '.join(required_keys)}")
    
    if required_keys and settings.is_development:
        print(f"⚠️  Warning: Missing API keys (development mode): {', '.join(required_keys)}")
        print(f"⚠️  The system will use fallback data instead of real API data")
    else:
        print(f"✅ All required API keys are present")

# Initialize and validate settings
settings = get_settings()
validate_api_keys(settings)