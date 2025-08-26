"""Custom exceptions for Heroku LangChain integration."""

from typing import Optional


class HerokuAPIError(Exception):
    """Base exception for Heroku API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class HerokuAuthenticationError(HerokuAPIError):
    """Authentication failed with Heroku API."""
    pass


class HerokuRateLimitError(HerokuAPIError):
    """Rate limit exceeded for Heroku API."""
    pass


class HerokuTimeoutError(HerokuAPIError):
    """Request timeout to Heroku API."""
    pass


class HerokuConfigurationError(HerokuAPIError):
    """Invalid configuration for Heroku client."""
    pass