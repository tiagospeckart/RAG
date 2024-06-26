from pydantic import BaseModel, Field, root_validator

from app.settings.settings_loader import load_settings


class CorsSettings(BaseModel):
    """CORS configuration.

    For more details on the CORS configuration, see:
    * https://fastapi.tiangolo.com/tutorial/cors/
    * https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
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
    """Server configuration.
    It's used by __main__.py
    """
    env_name: str = Field(
        description="Name of the environment (prod, staging, local...)"
    )
    port: int = Field(description="Port of FastAPI server, defaults to 8001")
    cors: CorsSettings = Field(
        description="CORS configuration", default=CorsSettings(enabled=False)
    )
    # hack for port type validation
    @root_validator(pre=True)
    def parse_port(cls, values):
        if 'port' in values and isinstance(values['port'], str):
            try:
                values['port'] = int(values['port'])  # Convert port string to int
            except ValueError:
                raise ValueError("Port must be a valid integer.")
        return values


class AzureOpenAISettings(BaseModel):
    api_key: str = Field(
        description="Secret key, shoulb be set in environment",
    )
    azure_endpoint: str = Field(
        description="Used to construct the Request endpoint",
    )
    azure_deployment: str = Field(
        description="Used to construct the Request endpoint",
    )
    api_version: str = Field(
        "2023_05_15",
        description="The API version to use for this operation. This follows the YYYY-MM-DD format.",
    )
    embedding_model: str = Field(
        "text-embedding-ada-002",
        description="OpenAI Model to use. Example: 'text-embedding-ada-002'.",
    )
    llm_model: str = Field(
        "gpt-35-turbo",
        description="OpenAI Model to use. Example: 'gpt-4'.",
    )
    temperature: str = Field(
        "0.1",
        description="Value used to manipulate the behaviour of the llm",
    )


# Update this Class with other Settings Classes as the project expands
class Settings(BaseModel):
    server: ServerSettings
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
