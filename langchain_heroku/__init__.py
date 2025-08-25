from importlib import metadata

from langchain_heroku.chat_models import ChatHeroku
from langchain_heroku.config import HerokuConfig
from langchain_heroku.embeddings import HerokuEmbeddings

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
    "__version__",
]
