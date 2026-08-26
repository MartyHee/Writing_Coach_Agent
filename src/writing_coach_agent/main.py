"""ASGI application exposed for Uvicorn and other process managers."""
from .config import Settings
from .web import create_app

settings = Settings.from_env()
app = create_app(settings)
