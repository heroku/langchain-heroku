"""Shared configuration for Heroku LangChain integrations."""

import os
from typing import Optional


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
            raise ValueError(f"{cls.INFERENCE_URL} must be set via env or init param.")
        if not cls.get_api_key(api_key):
            raise ValueError(f"{cls.INFERENCE_KEY} or {cls.INFERENCE_EMBED_KEY} must be set via env or init param.")
        if not cls.get_model_id(model_id):
            raise ValueError(f"model or {cls.INFERENCE_MODEL_ID} must be set via env or init param.")

    @classmethod
    def get_env(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value."""
        return os.environ.get(key, default)
