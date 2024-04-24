"""FastAPI app creation, logger configuration and main API routes."""
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from injector import Injector

from app import constants
from app.components.document_store import FAISSDocumentStore
from app.server.chat.chat_router import chat_router
from app.settings.settings import Settings

logger = logging.getLogger(__name__)


def create_app(root_injector: Injector) -> FastAPI:
    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        """
        Bind the root_injector to the state of incoming HTTP requests.

        This function is used as a dependency in FastAPI to ensure that the root_injector
        is available for use within the context of handling requests.

        Parameters:
        - request (Request): Represents the incoming HTTP request.

        Returns:
        - None
        """
        request.state.injector = root_injector

    # Fast API configuration + routing
    app = FastAPI(
        title="LangChain RAG Server",
        summary="Simple server to test and implement RAG with LLMs and LangChain to enhance the context of a query",
        version="1.0",
        dependencies=[Depends(bind_injector_to_request)])

    api_version_prefix = "/v1"
    app.include_router(chat_router, prefix=api_version_prefix)
    
    # Redirect to /docs
    @app.get("/")
    def redirect_to_docs():
        return RedirectResponse(url=api_version_prefix)
    
    settings = root_injector.get(Settings)

    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=settings.server.cors.allow_credentials,
            allow_origins=settings.server.cors.allow_origins,
            allow_origin_regex=settings.server.cors.allow_origin_regex,
            allow_methods=settings.server.cors.allow_methods,
            allow_headers=settings.server.cors.allow_headers,
        )

    return app
