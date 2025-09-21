# server/agents/base.py
"""
Base agent class for all AI agents in the system
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Generic, List
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add server directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.config import get_settings
from core.cache import CacheManager
from core.exceptions import AgentError, AgentConfigError

# Type variables for generic typing
RequestType = TypeVar('RequestType', bound=BaseModel)
ResponseType = TypeVar('ResponseType', bound=BaseModel)

class BaseAgent(ABC, Generic[RequestType, ResponseType]):
    """
    Base class for all AI agents
    
    Provides common functionality like:
    - Configuration management
    - Caching
    - Error handling
    - Logging
    - Rate limiting
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.settings = get_settings()
        self.config = self.settings.get_agent_config(agent_name)
        self.logger = logging.getLogger(f"agents.{agent_name}")
        self.cache = CacheManager() if self.settings.cache_enabled else None
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info(f"Initialized {agent_name} agent")
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate agent-specific configuration"""
        pass
    
    @abstractmethod
    async def process_request(self, request: RequestType) -> ResponseType:
        """Process agent request - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_fallback_response(self, request: RequestType, error: Exception) -> ResponseType:
        """Get fallback response when agent fails"""
        pass
    
    def get_cache_key(self, request: RequestType) -> str:
        """Generate cache key for request"""
        # Default implementation - can be overridden
        request_dict = request.dict() if hasattr(request, 'dict') else str(request)
        return f"{self.agent_name}:{hash(str(sorted(request_dict.items())))}"
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        if not self.cache:
            return None
        
        try:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                self.logger.info(f"Cache hit for key: {cache_key}")
                return cached_data
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
        
        return None
    
    async def set_cached_response(
        self, 
        cache_key: str, 
        response: ResponseType, 
        ttl: Optional[int] = None
    ) -> None:
        """Cache response"""
        if not self.cache:
            return
        
        try:
            ttl = ttl or self.settings.cache_default_ttl
            response_dict = response.dict() if hasattr(response, 'dict') else response
            await self.cache.set(cache_key, response_dict, ttl)
            self.logger.info(f"Cached response for key: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")
    
    async def execute(self, request: RequestType, use_cache: bool = True) -> ResponseType:
        """
        Main execution method with caching and error handling
        """
        start_time = datetime.now()
        cache_key = self.get_cache_key(request)
        
        try:
            # Check cache first
            if use_cache:
                cached_response = await self.get_cached_response(cache_key)
                if cached_response:
                    # Convert dict back to response model
                    response_class = self._get_response_class()
                    return response_class(**cached_response)
            
            # Process request
            self.logger.info(f"Processing {self.agent_name} request")
            response = await self.process_request(request)
            
            # Cache successful response
            if use_cache:
                await self.set_cached_response(cache_key, response)
            
            # Log execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Request processed in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            
            # Try to return fallback response
            try:
                fallback_response = self.get_fallback_response(request, e)
                self.logger.info("Returned fallback response")
                return fallback_response
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed: {fallback_error}")
                raise AgentError(f"{self.agent_name} agent failed: {e}") from e
    
    def _get_response_class(self) -> Type[ResponseType]:
        """Get response class for this agent - should be overridden if using generics"""
        # This is a fallback - agents should override this method
        return dict
    
    async def health_check(self) -> Dict[str, Any]:
        """Agent health check"""
        try:
            # Basic configuration check
            self._validate_config()
            
            return {
                "agent": self.agent_name,
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "config_valid": True,
                "cache_enabled": self.cache is not None
            }
        except Exception as e:
            return {
                "agent": self.agent_name,
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "config_valid": False
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.agent_name,
            "version": "1.0.0",
            "config": self.config,
            "cache_enabled": self.cache is not None,
            "description": self.__class__.__doc__ or f"{self.agent_name} agent"
        }

class AgentRegistry:
    """Registry for managing multiple agents"""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("agents.registry")
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent"""
        self._agents[agent.agent_name] = agent
        self.logger.info(f"Registered agent: {agent.agent_name}")
    
    def get(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self._agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self._agents.keys())
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Health check all agents"""
        results = {}
        for name, agent in self._agents.items():
            results[name] = await agent.health_check()
        return results
    
    def get_agents_info(self) -> Dict[str, Any]:
        """Get information about all agents"""
        return {
            name: agent.get_agent_info() 
            for name, agent in self._agents.items()
        }

# Global agent registry
agent_registry = AgentRegistry()