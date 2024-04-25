import logging
import os

from injector import singleton
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import utils as chromautils
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_weaviate import WeaviateVectorStore
import weaviate

logger = logging.getLogger(__name__)


@singleton
class ChromaDocumentStore:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Initialize the ChromaDocumentStore with specified chunking parameters and setup the Chroma vector database.

        Args:
            chunk_size (int): Size of each document chunk in characters for processing.
            chunk_overlap (int): Number of overlapping characters between consecutive document chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self. = self._initialize_()

    def _initialize_chroma_db(self) -> :
        """
        Initialize the Chroma vector database by loading and processing documents from a directory.

        Returns:
            Chroma: Initialized Chroma vector database.
        """
        logger.debug("Initializing Chroma Document store component=%s", type(self).__name__)

        # Load all Markdown files in the Documents Path as a List of Documents
        documents_path = constants.DOCUMENTS_PATH
        text_loader_kwargs = {'autodetect_encoding': True}
        loader = DirectoryLoader(documents_path, glob="**/*.md", loader_kwargs=text_loader_kwargs)
        documents = loader.load()

        # Split all Documents according to Chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        print("Doc Split Size : " + str(len(chunks)))
        print(chunks[0])

        # connect to running weaviate instance
        weaviate_client = weaviate.connect_to_local()
        vectorstore = WeaviateVectorStore.from_documents(documents=chunks,
                                                         embedding=self.embedding_function,
                                                         client=weaviate_client)

        # Create Chroma vector database injesting the Chunks with the chosen Embedding Function
        return vectorstore


conversation_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key="answer",
    input_key="question",
    k=4
)
chat_history = []
# vectorstore.persist()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
    )
system_instruction = "The assistant should provide detailed explanations. also use the Chat_History to impruve your answer. reply with short answers"
template = (
    f"system,{system_instruction} "
    "Combine the chat history , documents and follow up question into "
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
    # memory = chat_history
    get_chat_history=lambda h : h,
    memory=conversation_memory,
)
chain = load_qa_chain(model, chain_type="refine")

# print("ate aqui foi")
# response = retriever.get_relevant_documents("T-Store")
# print(response)
# print("aqui e apos o retriever funcionar")
# Examplo da Azure com qa

# answser = qa({"question": query, "chat_history": chat_history})
# print(answser)
# Aware_retriever

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
    model, retriever, contextualize_q_prompt
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
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
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
    return_source_documents=True,
)
