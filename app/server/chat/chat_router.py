from typing import Union, List

from fastapi import APIRouter, Depends
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from app.components.azure_openai_api_manager import SingletonAzureChat
from app.components.chroma_document_store import ChromaDocumentStore
from app.server.chat.chat_service import ChatService

chat_router = APIRouter()

# Instantiation of required objects
llm_component = SingletonAzureChat.get_instance()

chroma_doc_store = ChromaDocumentStore()

class Message(BaseModel):
    role: str
    content: str

class InputChat(BaseModel):
    messages: List[Union[Message]]


def get_chat_service() -> ChatService:
    return ChatService(llm_component, chroma_doc_store)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, professional assistant named Mag."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


@chat_router.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@chat_router.post("/chat")
async def chat_runnable(query: str, chat_service=Depends(get_chat_service)):
    chat_history = []
    answer = chat_service.query_chat(query, chat_history)
    
    return answer

@chat_router.post("/chat")
async def chat_runnable(query: str, chat_service=Depends(get_chat_service)):
    chat_history = []
    answer = chat_service.query_chat(query, chat_history)
    
    return answer