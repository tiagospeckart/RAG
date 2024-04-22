from typing import Union, List

from fastapi import APIRouter, Depends
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
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

### Rag Chain With History ###

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm_component, chroma_doc_store.chroma_db.as_retriever(), contextualize_q_prompt
)
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm_component, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@chat_router.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@chat_router.post("/chat")
async def chat_runnable(query: str, chat_service=Depends(get_chat_service)):
    chat_history = []
    answer = chat_service.query_chat(query, chat_history)
    
    return answer

@chat_router.get("/chatAware")
async def chat_aware(msg : str ,session_id):
    answer = conversational_rag_chain.invoke(
    {"input": msg},
    config={
        "configurable": {"session_id": session_id}
    },
    )["answer"]
    return answer