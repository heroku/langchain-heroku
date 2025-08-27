"""Custom exceptions for Heroku LangChain integration."""

from typing import Optional


class HerokuAPIError(Exception):
    """Base exception for Heroku API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[dict] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}
        self.request_id = request_id
        self.endpoint = endpoint

    def __str__(self) -> str:
        """Enhanced string representation for better debugging."""
        parts = [super().__str__()]

        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        if self.endpoint:
            parts.append(f"Endpoint: {self.endpoint}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        if self.response_data and isinstance(self.response_data, dict) and "error" in self.response_data:
            error_details = self.response_data["error"]
            if isinstance(error_details, dict):
                if "type" in error_details:
                    parts.append(f"Error Type: {error_details['type']}")
                if "code" in error_details:
                    parts.append(f"Error Code: {error_details['code']}")

        return " | ".join(parts)


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

    def __init__(self, message: str, config_field: Optional[str] = None, suggested_fix: Optional[str] = None):
        super().__init__(message)
        self.config_field = config_field
        self.suggested_fix = suggested_fix

    def __str__(self) -> str:
        """Enhanced configuration error message with helpful suggestions."""
        parts = [super().__str__()]

        if self.config_field:
            parts.append(f"Field: {self.config_field}")
        if self.suggested_fix:
            parts.append(f"Suggestion: {self.suggested_fix}")

        return " | ".join(parts)


class HerokuStreamingError(HerokuAPIError):
    """Error during streaming operations."""

    pass


class HerokuValidationError(HerokuAPIError):
    """Request validation error."""

    pass
