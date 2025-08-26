from importlib import metadata

from langchain_heroku.chat_models import ChatHeroku
from langchain_heroku.config import HerokuClientConfig, HerokuConfig
from langchain_heroku.embeddings import HerokuEmbeddings
from langchain_heroku.exceptions import (
    HerokuAPIError,
    HerokuAuthenticationError,
    HerokuConfigurationError,
    HerokuRateLimitError,
    HerokuTimeoutError,
)
from langchain_heroku.http_client import HerokuHTTPClient
from langchain_heroku.tool_converter import ToolConverter

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatHeroku",
    "HerokuEmbeddings",
    "HerokuConfig",
    "HerokuClientConfig",
    "HerokuAPIError",
    "HerokuAuthenticationError",
    "HerokuConfigurationError",
    "HerokuRateLimitError",
    "HerokuTimeoutError",
    "HerokuHTTPClient",
    "ToolConverter",
    "__version__",
]
