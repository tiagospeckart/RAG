import logging

from injector import singleton
from langchain_openai import AzureChatOpenAI

from app.settings.settings import settings

logger = logging.getLogger(__name__)

@singleton
class SingletonAzureChat:
    """
    SingletonAzureChat is a singleton class for managing a single instance of AzureChatOpenAI.
    
    This class ensures that only one instance of AzureChatOpenAI is created and used throughout 
    the application lifetime.
    """
    _instance = None

    def __new__(cls):
        """
        Create a new instance if one doesn't exist; otherwise, return the existing instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the singleton instance with AzureChatOpenAI if not already initialized.
        """
        if not self._initialized:
            logger.debug("Initializing Azure Openai API component=%s", type(self).__name__)
            self.model = AzureChatOpenAI(
                api_key=settings().azopenai.api_key,
                azure_endpoint=settings().azopenai.azure_endpoint,
                azure_deployment=settings().azopenai.azure_deployment,
                api_version=settings().azopenai.api_version
                )
            self._initialized = True

    @classmethod
    def get_instance(cls) -> AzureChatOpenAI:
        """
        Get the singleton instance of AzureChatOpenAI.
        
        Returns:
            AzureChatOpenAI: The singleton instance of AzureChatOpenAI.
        """
        return cls()._instance.model
