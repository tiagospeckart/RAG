from injector import singleton, inject
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from openai.types import Completion

from components.azure_openai_api_manager import get_azure_llm
from components.vector_store_component import VectorStoreComponent
from settings.settings import unsafe_typed_settings, Settings


@singleton
class ChatService:
    settings: Settings
    llm_component: AzureChatOpenAI
    vector_store_component: VectorStoreComponent

    @inject
    def __init__(
            self,
            settings: Settings,
            llm_component: AzureChatOpenAI,
            vector_store_component: VectorStoreComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.vector_store_component = vector_store_component

    def query_chat(
            self,
            query: str,
            chat_history: list
    ) -> dict:
        retriever = self.vector_store_component.get_vector_store().as_retriever()
        llm_component = get_azure_llm()
        system_instruction = "The assistant should provide detailed explanations."

        template = (
            f"{system_instruction} "
            "Combine the chat history and follow up question into "
            "a standalone question. Chat History: {chat_history} "
            "Follow up question: {question}"
        )

        condense_question_prompt = PromptTemplate.from_template(template)

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm_component,
            retriever=retriever,
            return_source_documents=True,
            condense_question_prompt=condense_question_prompt,
            chain_type="stuff",
        )

        conversation_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=2,
            return_messages=True
        )

        # Run your query
        result = qa.run(query, chat_history, conversation_memory)
        return result
