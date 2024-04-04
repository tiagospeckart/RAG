import os
from typing import List, Union

import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

# load environment variables
load_dotenv()

# Model configuration
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-03-15-preview"
openai.api_type = "azure"

model = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    temperature=0,
    openai_api_version="2023-05-15"
)

# FastAPI configuration
app = FastAPI(
    title="LangChain Server",
    version="0.1",
    description="Spin up a simple API server using Langchain's Runnable interfaces"
)

# Declare a chain + hardcoded system prompt
# TODO: refactor out to its proper place

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, professional assistant named Mag."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model


# classes
# TODO: refactor out to its proper place
class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation."
    )


# Routes
# TODO: put this code inside it's own module
path = "/v1"


@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.post(path + "/invoke")
async def invoke_runnable():
    pass


@app.post(path + "/invoke")
async def invoke_runnable():
    pass


@app.post(path + "/stream")
async def stream_runnable():
    pass


# Server init

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host=host, port=port)
