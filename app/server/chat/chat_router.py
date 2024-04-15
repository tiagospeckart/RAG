from typing import Union, List

from fastapi import APIRouter, FastAPI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

from app.constants import DOCUMENTS_PATH
from components.document_loader import load_docs
from components.document_splitter import split_docs

## TODO -> compare with server.py

chat_router = APIRouter(prefix="/v1")

retriever = vectorstore.as_retriever()
system_instruction = "The assistant should provide detailed explanations."
template = (
    f"{system_instruction} "
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)
condense_question_prompt = PromptTemplate.from_template(template)
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    return_source_documents=True,
    condense_question_prompt=condense_question_prompt,
    chain_type="stuff",
)
chain = load_qa_chain(model, chain_type="refine")
query = "O que é a T-Store?"

# print("ate aqui foi")
# response = retriever.get_relevant_documents("T-Store")
# print(response)
# print("aqui e apos o retriever funcionar")
# Examplo da Azure com qa
chat_history = []
result = {}
# answser = qa({"question": query, "chat_history": chat_history})
# print(answser)
# Conversario memory

conversation_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=2,
    return_messages=True
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
async def invoke_runnable2():
    pass


@app.post(path + "/stream")
async def stream_runnable():
    pass


@app.get(path + "/chat")
async def chat_runnable(msg: str):
    answser = qa({"question": msg, "chat_history": chat_history})
    return answser
