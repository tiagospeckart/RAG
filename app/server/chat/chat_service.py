from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

from components.azure_openai_api_manager import get_azure_llm
from components.vector_store_component import VectorStoreComponent
from settings.settings import settings

vector_store_instance = VectorStoreComponent(settings())
retriever = vector_store_instance.vector_store.as_retriever()

system_instruction = "The assistant should provide detailed explanations."

template = (
    f"{system_instruction} "
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)

condense_question_prompt = PromptTemplate.from_template(template)

qa = ConversationalRetrievalChain.from_llm(
    llm=get_azure_llm(),
    retriever=retriever,
    return_source_documents=True,
    condense_question_prompt=condense_question_prompt,
    chain_type="stuff",
)

query = "O que é a T-Store?"
chat_history = []

result = {}
conversation_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=2,
    return_messages=True
)