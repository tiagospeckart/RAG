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
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import utils as chromautils
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
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

# Start DocLoader
directory = './wiki-single-file.md'

def load_docs(directory):
  loader = UnstructuredMarkdownLoader(directory, mode = "elements")
  documents = loader.load()
  return documents

documents = load_docs(directory)

def split_docs(documents,chunk_size=1000,chunk_overlap=20,
    length_function = len,):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print("Doc Split Size : "+str(len(docs)))
print(docs[0])
# End DocLoader
# --------------------//--------------------//--------------------//--------------------
# Start Embbedings and ChromaDB

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# print(embeddings.embed_documents)
docschroma = chromautils.filter_complex_metadata(docs)

vectorstore = Chroma.from_documents(documents=docs,
                                    embedding=embeddings,
                                    persist_directory="./db"
                                    )

vectorstore.persist()
retriever = vectorstore.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm = model,
    retriever=retriever,
    return_source_documents=True,
)

chain = load_qa_chain(model, chain_type="refine")
query = "What's T-Store?"
chain.run(input_documents=docs, question=query)

# Conversario memory

conversation_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
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


# Server init

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host=host, port=port)
