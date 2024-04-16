import logging
import typing

from injector import singleton, inject
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.vectorstores import VectorStore

from constants import DB_PATH
from settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class VectorStoreComponent:
    settings: Settings
    vector_store: VectorStore

    @inject
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        match settings.vectorstore.database:
            case "chroma":
                try:
                    import chromadb  # type: ignore
                    from chromadb.config import (  # type: ignore
                        Settings as ChromaSettings,
                    )

                except ImportError as e:
                    raise ImportError(
                        "ChromaDB dependencies not found, install with `poetry install --extras vector-stores-chroma`"
                    ) from e

                chroma_settings = ChromaSettings(anonymized_telemetry=False)
                chroma_client = chromadb.PersistentClient(
                    path=str(DB_PATH.absolute()),
                    settings=chroma_settings,
                )
                chroma_client.get_or_create_collection(
                    "make_this_parameterizable_per_api_call"
                )

                self.vector_store = typing.cast(VectorStore, Chroma)

    def get_vector_store(self) -> "VectorStoreComponent":
        return self
