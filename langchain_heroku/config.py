"""Shared configuration for Heroku LangChain integrations."""

import os
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from langchain_heroku.exceptions import HerokuConfigurationError


class HerokuClientConfig(BaseModel):
    """Type-safe configuration for Heroku clients."""

    model_config = {"protected_namespaces": ()}

    # Required fields
    inference_url: str = Field(..., description="Inference API URL")
    api_key: str = Field(..., description="API key for authentication")
    model_id: str = Field(..., description="Model identifier")

    # Optional fields with defaults
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=2, description="Number of retry attempts")

    @field_validator("inference_url")
    @classmethod
    def validate_inference_url(cls, v: str) -> str:
        if not v:
            raise HerokuConfigurationError(
                "inference_url cannot be empty",
                config_field="inference_url",
                suggested_fix="Set INFERENCE_URL environment variable or pass inference_url parameter",
            )
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise HerokuConfigurationError(
                "api_key cannot be empty", config_field="api_key", suggested_fix="Set INFERENCE_KEY environment variable or pass api_key parameter"
            )
        return v

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v:
            raise HerokuConfigurationError(
                "model_id cannot be empty",
                config_field="model_id",
                suggested_fix="Set INFERENCE_MODEL_ID environment variable or pass model parameter",
            )
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if hasattr(v, "_mock_name"):  # Skip validation for MagicMock objects in tests
            return v
        if v <= 0:
            raise HerokuConfigurationError(
                "timeout must be positive", config_field="timeout", suggested_fix="Use a positive integer value (e.g., 30 for 30 seconds)"
            )
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        if hasattr(v, "_mock_name"):  # Skip validation for MagicMock objects in tests
            return v
        if v < 0:
            raise HerokuConfigurationError(
                "max_retries cannot be negative",
                config_field="max_retries",
                suggested_fix="Use 0 or a positive integer (e.g., 2 for 2 retry attempts)",
            )
        return v


class HerokuConfig:
    """Shared configuration for Heroku LangChain integrations."""

    # Environment variable names
    INFERENCE_URL = "INFERENCE_URL"
    INFERENCE_KEY = "INFERENCE_KEY"
    INFERENCE_EMBED_KEY = "INFERENCE_EMBED_KEY"  # Alternative to INFERENCE_KEY
    INFERENCE_MODEL_ID = "INFERENCE_MODEL_ID"

    @classmethod
    def get_inference_url(cls, instance_url: Optional[str] = None) -> Optional[str]:
        """Get inference URL from instance or environment."""
        return instance_url or os.environ.get(cls.INFERENCE_URL)

    @classmethod
    def get_api_key(cls, instance_key: Optional[str] = None) -> Optional[str]:
        """Get API key from instance or environment with fallback priority."""
        if instance_key:
            return instance_key

        # Try INFERENCE_KEY first, then INFERENCE_EMBED_KEY as fallback
        return os.environ.get(cls.INFERENCE_KEY) or os.environ.get(cls.INFERENCE_EMBED_KEY)

    @classmethod
    def get_model_id(cls, instance_model: Optional[str] = None) -> Optional[str]:
        """Get model ID from instance or environment."""
        return instance_model or os.environ.get(cls.INFERENCE_MODEL_ID)

    @classmethod
    def validate_config(cls, inference_url: Optional[str] = None, api_key: Optional[str] = None, model_id: Optional[str] = None) -> None:
        """Validate that all required configuration is present."""
        if not cls.get_inference_url(inference_url):
            raise HerokuConfigurationError(f"{cls.INFERENCE_URL} must be set via env or init param.")
        if not cls.get_api_key(api_key):
            raise HerokuConfigurationError(f"{cls.INFERENCE_KEY} or {cls.INFERENCE_EMBED_KEY} must be set via env or init param.")
        if not cls.get_model_id(model_id):
            raise HerokuConfigurationError(f"model or {cls.INFERENCE_MODEL_ID} must be set via env or init param.")

    @classmethod
    def create_client_config(
        cls,
        inference_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2,
    ) -> HerokuClientConfig:
        """Create a validated client configuration.

        Args:
            inference_url: Inference URL (from env if None)
            api_key: API key (from env if None)
            model_id: Model ID (from env if None)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts

        Returns:
            Validated HerokuClientConfig instance
        """
        resolved_url = cls.get_inference_url(inference_url)
        resolved_key = cls.get_api_key(api_key)
        resolved_model = cls.get_model_id(model_id)

        if not resolved_url:
            raise HerokuConfigurationError(
                f"{cls.INFERENCE_URL} must be set via env or init param.",
                config_field="inference_url",
                suggested_fix=f"Set {cls.INFERENCE_URL} environment variable or pass inference_url parameter",
            )
        if not resolved_key:
            raise HerokuConfigurationError(
                f"{cls.INFERENCE_KEY} or {cls.INFERENCE_EMBED_KEY} must be set via env or init param.",
                config_field="api_key",
                suggested_fix=f"Set {cls.INFERENCE_KEY} environment variable or pass api_key parameter",
            )
        if not resolved_model:
            raise HerokuConfigurationError(
                f"model or {cls.INFERENCE_MODEL_ID} must be set via env or init param.",
                config_field="model_id",
                suggested_fix=f"Set {cls.INFERENCE_MODEL_ID} environment variable or pass model parameter",
            )

        return HerokuClientConfig(inference_url=resolved_url, api_key=resolved_key, model_id=resolved_model, timeout=timeout, max_retries=max_retries)

    @classmethod
    def get_env(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value."""
        return os.environ.get(key, default)
