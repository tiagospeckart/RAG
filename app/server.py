import os
from typing import List, Union

import httpx
from langchain_weaviate import WeaviateVectorStore
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import utils as chromautils
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.vectorstores.weaviate import Weaviate
import weaviate

# load environment variables
load_dotenv()

# Model configuration
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"

model = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    temperature=0.08,
    openai_api_version="2023-05-15",
    http_client=httpx.Client(verify=False)
)

# Start DocLoader
# directory = './files/wiki-single-file.md'
# remember to create a folder named files and copy your {document_name}.md
documents_folder = os.getenv("DOCUMENTS_FOLDER")
chroma_folder = os.getenv("DB_FOLDER")


def load_docs(docs_path):
    #   loader = UnstructuredMarkdownLoader(directory, mode = "single")
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(docs_path, glob="./*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    loaded_documents = loader.load()
    return loaded_documents


documents = load_docs(documents_folder)


def split_docs(unsplitted_docs,
               chunk_size=1000,
               chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_docs = text_splitter.split_documents(unsplitted_docs)
    return splitted_docs


chunks = split_docs(documents)
print("Doc Split Size : " + str(len(chunks)))
print(chunks[0])

# End DocLoader
# --------------------//--------------------//--------------------//--------------------
# Start Embbedings and ChromaDB

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# print(embeddings.embed_documents)
docschroma = chromautils.filter_complex_metadata(chunks)
# print(docschroma[0])
# vectorstore = Chroma.from_documents(documents=docschroma,
#                                     embedding=embeddings,
#                                     # persist_directory=chroma_folder
#                                     )
# # Conversario memory

client = weaviate.connect_to_local()
vectorstore = WeaviateVectorStore.from_documents(documents=chunks,
                                                 embedding=embeddings,
                                                 client=client)
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
# Aware Retriever end
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
            "examples":{ 
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
async def chat_runnable(msg : str ):
    # answer = qa.invoke({"question": msg})
    answer = qa.invoke({"question": msg, "chat_history": conversation_memory})
    # answer = qa.invoke({"question": msg, "chat_history": chat_history})
    # chat_history.extend([HumanMessage(content=msg), answer["answer"]])
    return answer

@app.get(path + "/chatAware")
async def chatAware(msg : str ,session_id):
    answer = conversational_rag_chain.invoke(
    {"input": msg},
    config={
        "configurable": {"session_id": session_id}
    },
    )["answer"]
    return answer


# Server init

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host=host, port=port)
