import httpx
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()


# Singleton Class
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
                http_client=httpx.Client(verify=False)
            )
            self._initialized = True


def get_azure_llm() -> AzureChatOpenAI:
    singleton = SingletonAzureChat()
    return singleton.model
