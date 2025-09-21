# server/core/cache.py
"""
Caching utilities
"""
import asyncio
from typing import Any, Optional, Dict
import json
import time
from cachetools import TTLCache

class CacheManager:
    """Simple in-memory cache manager"""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Any] = TTLCache(maxsize=max_size, ttl=900)  # 15 min default
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 900) -> None:
        """Set value in cache with TTL"""
        # For TTLCache, we can't set individual TTLs easily
        # This is a simplified implementation
        self._cache[key] = value
    
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        self._cache.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache"""
        self._cache.clear()