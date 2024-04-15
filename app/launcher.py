"""FastAPI app creation, logger configuration and main API routes."""
import logging
import os

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector
from langchain_community.vectorstores.chroma import Chroma

from app.server.chat.chat_router import chat_router
from app.settings.settings import Settings
from components import document_loader, vector_db_manager, azure_openai_api_manager, document_splitter
from components.vector_db_manager import VectorDatabaseManager
from constants import DOCUMENTS_PATH, DB_PATH

logger = logging.getLogger(__name__)


def create_app(root_injector: Injector) -> FastAPI:

    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        request.state.injector = root_injector

    app = FastAPI(title="LangChain RAG Server",
                  version="0.2",
                  description="Spin up a simple API server using Langchain's Runnable interfaces",
                  dependencies=[Depends(bind_injector_to_request)])

    # TODO: add routes using "app.include_router(route)" later
    app.include_router(chat_router)

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

    documents = document_loader.load_docs(DOCUMENTS_PATH)
    splited_docs = document_splitter.split_docs(documents)

    # Initialize vector database
    persist_directory = str(DB_PATH)
    vectordb_manager = VectorDatabaseManager(persist_directory)
    vectordb_manager.load_or_create_vector_database(splited_docs, persist_directory)

    return app
