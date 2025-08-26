"""Shared configuration for Heroku LangChain integrations."""

import os
from dataclasses import dataclass, field
from typing import Optional

from langchain_heroku.exceptions import HerokuConfigurationError


@dataclass
class HerokuClientConfig:
    """Type-safe configuration for Heroku clients."""
    
    # Required fields
    inference_url: str
    api_key: str
    model_id: str
    
    # Optional fields with defaults
    timeout: int = field(default=30)
    max_retries: int = field(default=2)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.inference_url:
            raise HerokuConfigurationError("inference_url cannot be empty")
        if not self.api_key:
            raise HerokuConfigurationError("api_key cannot be empty")
        if not self.model_id:
            raise HerokuConfigurationError("model_id cannot be empty")
        if self.timeout <= 0:
            raise HerokuConfigurationError("timeout must be positive")
        if self.max_retries < 0:
            raise HerokuConfigurationError("max_retries cannot be negative")


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
        max_retries: int = 2
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
            raise HerokuConfigurationError(f"{cls.INFERENCE_URL} must be set via env or init param.")
        if not resolved_key:
            raise HerokuConfigurationError(f"{cls.INFERENCE_KEY} or {cls.INFERENCE_EMBED_KEY} must be set via env or init param.")
        if not resolved_model:
            raise HerokuConfigurationError(f"model or {cls.INFERENCE_MODEL_ID} must be set via env or init param.")
        
        return HerokuClientConfig(
            inference_url=resolved_url,
            api_key=resolved_key,
            model_id=resolved_model,
            timeout=timeout,
            max_retries=max_retries
        )

    @classmethod
    def get_env(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value."""
        return os.environ.get(key, default)
