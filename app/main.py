"""FastAPI app creation, logger configuration and main API routes."""

from app.di import global_injector
from app.launcher import create_app


app = create_app(global_injector)




