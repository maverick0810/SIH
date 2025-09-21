# server/core/exceptions.py
"""
Custom exceptions for the backend
"""

class KisanSathiError(Exception):
    """Base exception for KisanSathi backend"""
    pass

class AgentError(KisanSathiError):
    """Agent-related errors"""
    pass

class AgentConfigError(KisanSathiError):
    """Agent configuration errors"""
    pass

class ExternalAPIError(KisanSathiError):
    """External API errors"""
    pass

class CacheError(KisanSathiError):
    """Cache-related errors"""
    pass