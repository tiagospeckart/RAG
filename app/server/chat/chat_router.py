from typing import List, Union

from fastapi import APIRouter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

chat_router = APIRouter()

# TODO: prompt and chain declaration should be inside a service file

llm = AzureChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, professional assistant named Mag."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm


# classes
# TODO: refactor out to its proper place
class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation."
    )


@chat_router.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@chat_router.post("/invoke")
async def invoke_runnable():
    pass


@chat_router.post("/stream")
async def stream_runnable():
    pass
