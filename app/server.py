import os
from typing import List, Union

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

# load environment variables
load_dotenv()

# Model configuration
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-03-15-preview"
openai.api_type = "azure"

model = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    temperature=0.1,
    openai_api_version="2023-05-15"
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
vectorstore = Chroma.from_documents(documents=chunks,
                                    embedding=embeddings,
                                    persist_directory=chroma_folder
                                    )

vectorstore.persist()
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

print("ate aqui foi")
response = retriever.get_relevant_documents("T-Store")
print(response)
print("aqui e apos o retriever funcionar")
# Examplo da Azure com qa
chat_history = []
result = {}
answser = qa({"question": query, "chat_history": chat_history})
chat_history.append((query, answser))
print(answser)
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

# chain = prompt | model


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
