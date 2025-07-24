from importlib import metadata

from langchain_heroku.chat_models import ChatHeroku

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatHeroku",
    "__version__",
]
