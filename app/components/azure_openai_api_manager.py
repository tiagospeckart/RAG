from langchain_openai import AzureChatOpenAI
from injector import singleton

from app.settings.settings import settings


@singleton
class SingletonAzureChat:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.model = AzureChatOpenAI(
                api_key=settings().azopenai.api_key,
                azure_endpoint=settings().azopenai.azure_endpoint,
                azure_deployment=settings().azopenai.azure_deployment
                )
            self._initialized = True

    @classmethod
    def get_instance(cls) -> AzureChatOpenAI:
        return cls()._instance.model
