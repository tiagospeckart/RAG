from pydantic import BaseModel, Field

from app.settings.settings_loader import load_settings


class CorsSettings(BaseModel):
    """CORS configuration.

    For more details on the CORS configuration, see:
    # * https://fastapi.tiangolo.com/tutorial/cors/
    # * https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
    """

    enabled: bool = Field(
        description="Flag indicating if CORS headers are set or not."
                    "If set to True, the CORS headers will be set to allow all origins, methods and headers.",
        default=False,
    )
    allow_credentials: bool = Field(
        description="Indicate that cookies should be supported for cross-origin requests",
        default=False,
    )
    allow_origins: list[str] = Field(
        description="A list of origins that should be permitted to make cross-origin requests.",
        default=[],
    )
    allow_origin_regex: list[str] = Field(
        description="A regex string to match against origins that should be permitted to make cross-origin requests.",
        default=None,
    )
    allow_methods: list[str] = Field(
        description="A list of HTTP methods that should be allowed for cross-origin requests.",
        default=[
            "GET",
        ],
    )
    allow_headers: list[str] = Field(
        description="A list of HTTP request headers that should be supported for cross-origin requests.",
        default=[],
    )


class ServerSettings(BaseModel):
    env_name: str = Field(
        description="Name of the environment (prod, staging, local...)"
    )
    port: int = Field(description="Port of PrivateGPT FastAPI server, defaults to 8001")
    cors: CorsSettings = Field(
        description="CORS configuration", default=CorsSettings(enabled=False)
    )


class DataSettings(BaseModel):
    local_data_folder: str = Field(
        description="Path to local storage."
                    "It will be treated as an absolute path if it starts with /"
    )


class AzureOpenAISettings(BaseModel):
    api_key: str
    azure_endpoint: str
    api_version: str = Field(
        "2023_05_15",
        description="The API version to use for this operation. This follows the YYYY-MM-DD format.",
    )
    embedding_deployment_name: str
    embedding_model: str = Field(
        "text-embedding-ada-002",
        description="OpenAI Model to use. Example: 'text-embedding-ada-002'.",
    )
    llm_deployment_name: str
    llm_model: str = Field(
        "gpt-35-turbo",
        description="OpenAI Model to use. Example: 'gpt-4'.",
    )


class RerankSettings(BaseModel):
    enabled: bool = Field(
        False,
        description="This value controls whether a reranker should be included in the RAG pipeline.",
    )
    model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-2-v2",
        description="Rerank model to use. Limited to SentenceTransformer cross-encoder models.",
    )
    top_n: int = Field(
        2,
        description="This value controls the number of documents returned by the RAG pipeline.",
    )


class RagSettings(BaseModel):
    similarity_top_k: int = Field(
        2,
        description="This value controls the number of documents returned by the RAG pipeline or considered for "
                    "reranking if enabled.",
    )
    similarity_value: float = Field(
        None,
        description="If set, any documents retrieved from the RAG must meet a certain match score. Acceptable values "
                    "are between 0 and 1.",
    )
    rerank: RerankSettings


class Settings(BaseModel):
    server: ServerSettings
    data: DataSettings
    azopenai: AzureOpenAISettings


"""
This is visible just for DI or testing purposes.

Use dependency injection or `settings()` method instead.
"""
unsafe_settings = load_settings()

"""
This is visible just for DI or testing purposes.

Use dependency injection or `settings()` method instead.
"""
unsafe_typed_settings = Settings(**unsafe_settings)


def settings() -> Settings:
    """Get the current loaded settings from the DI container.

    This method exists to keep compatibility with the existing code,
    that require global access to the settings.

    For regular components use dependency injection instead.
    """
    from app.di import global_injector

    return global_injector.get(Settings)
