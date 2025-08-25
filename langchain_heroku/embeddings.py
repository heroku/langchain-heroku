"""Heroku embeddings models."""

from typing import Any, Dict, List, Optional, Union

import httpx
from langchain_core.embeddings import Embeddings

from langchain_heroku.config import HerokuConfig


class HerokuEmbeddings(Embeddings):
    """
    Heroku embeddings model integration using the Inference API v1 /v1/embeddings endpoint.

    This class provides a LangChain-compatible interface to Heroku's embeddings API,
    with careful handling of OpenAI compatibility issues as documented in EMBEDDINGS_NOTES.md.

    Example setup (environment variables):
        export INFERENCE_URL="https://your-inference-api-url"
        export INFERENCE_KEY="your-heroku-inference-api-key"
        export INFERENCE_MODEL_ID="your-model-id"

    Basic usage:
        from langchain_heroku.embeddings import HerokuEmbeddings

        embeddings = HerokuEmbeddings()
        result = embeddings.embed_query("Hello, world!")
        print(len(result))  # Vector dimension

    Batch usage:
        texts = ["Hello", "World", "How are you?"]
        results = embeddings.embed_documents(texts)
        print(f"Generated {len(results)} embeddings")

    Advanced usage with Heroku-specific features:
        embeddings = HerokuEmbeddings(
            model="cohere-embed-multilingual",
            input_type="search_document",
            encoding_format="raw",
            embedding_type="float",
            allow_ignored_params=True
        )
    """

    model: Optional[str] = None
    api_key: Optional[str] = None
    inference_url: Optional[str] = None
    timeout: Optional[int] = None

    # Heroku-specific parameters (not OpenAI compatible)
    input_type: Optional[str] = None  # "search_document", "search_query", "classification", "clustering"
    encoding_format: Optional[str] = None  # "raw" or "base64"
    embedding_type: Optional[str] = None  # "float", "int8", "uint8", "binary", "ubinary"
    allow_ignored_params: Optional[bool] = None  # Whether to ignore unsupported parameters

    def __init__(self, **kwargs: Any) -> None:
        """Initialize HerokuEmbeddings with configuration."""
        # Extract parameters from kwargs
        self.model = kwargs.get("model")
        self.api_key = kwargs.get("api_key")
        self.inference_url = kwargs.get("inference_url")
        self.timeout = kwargs.get("timeout")
        self.input_type = kwargs.get("input_type")
        self.encoding_format = kwargs.get("encoding_format")
        self.embedding_type = kwargs.get("embedding_type")
        self.allow_ignored_params = kwargs.get("allow_ignored_params")

        # Set defaults for OpenAI compatibility
        if self.encoding_format is None:
            self.encoding_format = "raw"
        if self.embedding_type is None:
            self.embedding_type = "float"
        if self.allow_ignored_params is None:
            self.allow_ignored_params = True

    @property
    def _llm_type(self) -> str:
        return "heroku_embeddings"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get parameters that identify this model."""
        return {
            "model": self.model,
            "input_type": self.input_type,
            "encoding_format": self.encoding_format,
            "embedding_type": self.embedding_type,
        }

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value."""
        return HerokuConfig.get_env(key, default)

    def _get_api_key(self) -> Optional[str]:
        """Get API key from instance or environment."""
        return HerokuConfig.get_api_key(self.api_key)

    def _get_inference_url(self) -> Optional[str]:
        """Get inference URL from instance or environment."""
        return HerokuConfig.get_inference_url(self.inference_url)

    def _get_model(self) -> Optional[str]:
        """Get model ID from instance or environment."""
        return HerokuConfig.get_model_id(self.model)

    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        HerokuConfig.validate_config(inference_url=self.inference_url, api_key=self.api_key, model_id=self.model)

    def _validate_input_type(self, input_type: Optional[str]) -> None:
        """Validate input_type parameter if provided."""
        if input_type is not None:
            valid_types = ["search_document", "search_query", "classification", "clustering"]
            if input_type not in valid_types:
                raise ValueError(f"input_type must be one of {valid_types}, got {input_type}")

    def _validate_encoding_format(self, encoding_format: Optional[str]) -> None:
        """Validate encoding_format parameter if provided."""
        if encoding_format is not None:
            valid_formats = ["raw", "base64"]
            if encoding_format not in valid_formats:
                raise ValueError(f"encoding_format must be one of {valid_formats}, got {encoding_format}")

    def _validate_embedding_type(self, embedding_type: Optional[str]) -> None:
        """Validate embedding_type parameter if provided."""
        if embedding_type is not None:
            valid_types = ["float", "int8", "uint8", "binary", "ubinary"]
            if embedding_type not in valid_types:
                raise ValueError(f"embedding_type must be one of {valid_types}, got {embedding_type}")

    def _build_payload(self, input_text: Union[str, List[str]]) -> Dict[str, Any]:
        """Build the API payload for the embeddings request."""
        payload: Dict[str, Any] = {
            "model": self._get_model(),
            "input": input_text,
        }

        # Add Heroku-specific parameters if they are set
        if self.input_type is not None:
            payload["input_type"] = self.input_type
        if self.encoding_format is not None:
            payload["encoding_format"] = self.encoding_format
        if self.embedding_type is not None:
            payload["embedding_type"] = self.embedding_type
        if self.allow_ignored_params is not None:
            payload["allow_ignored_params"] = self.allow_ignored_params

        return payload

    def _make_api_request(self, payload: dict) -> dict:
        """Make the API request with retry logic."""
        url = self._get_inference_url()
        api_key = self._get_api_key()
        timeout = self.timeout or 30
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        max_retries = 2
        for _ in range(max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(f"{url}/v1/embeddings", json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:
                last_exc = e
        else:
            raise RuntimeError(f"Heroku Embeddings API call failed after {max_retries} retries: {last_exc}")

    def _extract_embeddings(self, response: dict) -> List[List[float]]:
        """Extract embeddings from the API response."""
        try:
            data = response.get("data", [])

            # Check if data is empty
            if not data:
                raise ValueError("No embeddings returned from API")

            embeddings = []

            for item in data:
                if "embedding" not in item:
                    raise ValueError(f"Missing 'embedding' key in response item: {item}")

                embedding = item["embedding"]
                # Ensure we return float embeddings for LangChain compatibility
                if isinstance(embedding, list):
                    # Convert to float if needed (e.g., for int8/uint8 responses)
                    float_embedding = [float(x) for x in embedding]
                    embeddings.append(float_embedding)
                else:
                    raise ValueError(f"Unexpected embedding format: {type(embedding)}")

            return embeddings
        except (KeyError, TypeError) as e:
            raise ValueError(f"Failed to extract embeddings from response: {e}")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        self._validate_config()
        self._validate_input_type(self.input_type)
        self._validate_encoding_format(self.encoding_format)
        self._validate_embedding_type(self.embedding_type)

        payload = self._build_payload(text)
        response = self._make_api_request(payload)
        embeddings = self._extract_embeddings(response)

        if not embeddings:
            raise ValueError("No embeddings returned from API")

        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        self._validate_config()
        self._validate_input_type(self.input_type)
        self._validate_encoding_format(self.encoding_format)
        self._validate_embedding_type(self.embedding_type)

        payload = self._build_payload(texts)
        response = self._make_api_request(payload)
        return self._extract_embeddings(response)

    def embed_query_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Embed a single query text and return with metadata.

        This method provides access to additional Heroku API response data
        that may not be available in the standard OpenAI-compatible interface.
        """
        self._validate_config()
        self._validate_input_type(self.input_type)
        self._validate_encoding_format(self.encoding_format)
        self._validate_embedding_type(self.embedding_type)

        payload = self._build_payload(text)
        response = self._make_api_request(payload)

        embeddings = self._extract_embeddings(response)
        if not embeddings:
            raise ValueError("No embeddings returned from API")

        return {
            "embedding": embeddings[0],
            "model": response.get("model"),
            "usage": response.get("usage"),
            "response_metadata": response,
        }

    def embed_documents_with_metadata(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents and return with metadata.

        This method provides access to additional Heroku API response data
        that may not be available in the standard OpenAI-compatible interface.
        """
        self._validate_config()
        self._validate_input_type(self.input_type)
        self._validate_encoding_format(self.encoding_format)
        self._validate_embedding_type(self.embedding_type)

        payload = self._build_payload(texts)
        response = self._make_api_request(payload)

        embeddings = self._extract_embeddings(response)
        if len(embeddings) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")

        results = []
        for i, embedding in enumerate(embeddings):
            results.append(
                {
                    "embedding": embedding,
                    "text": texts[i],
                    "model": response.get("model"),
                    "usage": response.get("usage"),
                    "response_metadata": response,
                }
            )

        return results
