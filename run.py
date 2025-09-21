# server/run.py
"""
Main entry point for KisanSathi AI Backend - Updated with Irrigation Agent
"""

import sys
import os

# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed")

import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager

from api.app import create_app
from core.config import get_settings
from core.logging import setup_logging
from agents.mandi.agent import MandiAgent
from agents.irrigation.agent import IrrigationAgent
from agents.base import agent_registry

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    """Application lifespan management"""
    
    # Startup
    logger.info("🚀 Starting KisanSathi AI Backend")
    
    # Initialize and register agents
    logger.info("Initializing agents...")
    try:
        # Initialize Mandi Agent
        mandi_agent = MandiAgent()
        agent_registry.register(mandi_agent)
        logger.info("✅ Mandi agent registered")
        
        # Initialize Irrigation Agent
        irrigation_agent = IrrigationAgent()
        agent_registry.register(irrigation_agent)
        logger.info("✅ Irrigation agent registered")
        
        # Test agent health
        health_results = await agent_registry.health_check_all()
        for agent_name, health in health_results.items():
            status = "✅" if health["status"] == "healthy" else "❌"
            logger.info(f"{status} {agent_name}: {health['status']}")
        
        logger.info("🎯 All agents initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agents: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down KisanSathi AI Backend")

def create_application():
    """Create FastAPI application with all configurations"""
    return create_app(lifespan=lifespan)

def main():
    """Main entry point"""
    settings = get_settings()
    
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Run server with proper import string for reload
    if settings.debug:
        # Use import string for reload to work
        uvicorn.run(
            "run:create_application",
            factory=True,
            host=settings.api_host,
            port=settings.api_port,
            reload=True,
            log_level=settings.log_level.lower(),
            access_log=True
        )
    else:
        # Production mode - create app directly
        app = create_application()
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            reload=False,
            log_level=settings.log_level.lower(),
            access_log=True
        )

if __name__ == "__main__":
    main()