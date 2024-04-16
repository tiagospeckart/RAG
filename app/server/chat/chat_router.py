from typing import Union, List

from fastapi import APIRouter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ChatMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

from server.chat.chat_service import ChatService

chat_router = APIRouter(prefix="/v1")
service = ChatService

class Message(BaseModel):
    role: str
    content: str
    model_config = {
        "json_schema_extra": {
            "examples": {
                "role": "user",
                "content": "What is T-Store?"
            }
        }
    }


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation."
    )


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, professional assistant named Mag."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

@chat_router.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@chat_router.get("/chat")
async def chat_runnable(service: ChatService, msg: str):
    ChatService.query_chat(service, query=msg, chat_history=[])
