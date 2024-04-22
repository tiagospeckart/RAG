from injector import singleton, inject
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from app.components.azure_openai_api_manager import SingletonAzureChat
from app.components.chroma_document_store import ChromaDocumentStore


@singleton
class ChatService:
    llm_component: AzureChatOpenAI
    chroma_doc_store: ChromaDocumentStore

    @inject
    def __init__(
        self,
        llm_component: AzureChatOpenAI,
        chroma_doc_store: ChromaDocumentStore
    ) -> None:
        self.llm_component = llm_component
        self.chroma_doc_store = chroma_doc_store

    def query_chat(
        self,
        query: str,
        chat_history: list
    ) -> dict:
        chroma_db = self.chroma_doc_store.chroma_db
        retriever = chroma_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        llm_component = self.llm_component
        system_instruction = "The assistant should provide detailed explanations."

        template = (
            f"{system_instruction} "
            "Combine the chat history and follow up question into "
            "a standalone question. Chat History: {chat_history} "
            "Follow up question: {query}"
        )
        
        conversation_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key="answer",
            input_key="question",
            k=4
        )

        condense_question_prompt = PromptTemplate.from_template(template)

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm_component,
            retriever=retriever,
            condense_question_prompt=condense_question_prompt,
            chain_type="stuff",
            get_chat_history=lambda h : h,
            memory=conversation_memory
        )

        return qa({"question": query, "chat_history": chat_history})


