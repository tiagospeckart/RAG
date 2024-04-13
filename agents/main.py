import os
import vector_database
import text_splitter
import document_loader
import azure_openai_api
import tool_manager
import agent_manager

# from langchain.agents
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

def test_llm():
    question = "What is T-Store?"
    answer = azure_openai_api.ask_documents(llm, question, documents)
    print("\nAsk documents: " + answer)

    chat = azure_openai_api.create_chat(llm, vectordb)

    answer = azure_openai_api.ask_vectordb(chat, question)
    print("\nAsk vectordb: " + answer['answer'])

    # question = "What was my last question?"
    # answer = azure_openai_api.ask_vectordb(chat, question)
    # print("\nAsk vectordb: " + answer['answer'])

def test_single_agent():
    tools = tool_manager.get_tools()

    agent = agent_manager.create_agent_executor(llm, tools)

    query = "What is the current user id?"
    # agent.run(query)
    agent.invoke({"input": query})


if __name__ == "__main__":
    load_dotenv("azure.env")
    
    documents = document_loader.load_docs()
    splited_docs = text_splitter.split_docs(documents)
    
    persist_directory = os.getenv("VECTOR_DATABASE_FOLDER")
    ## TODO Check for documents changes
    if os.path.exists(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=vector_database.generate_embeddings())
    else:
        vectordb = vector_database.create_vector_database(splited_docs, persist_directory)
    
    llm = azure_openai_api.create_llm()
    
    test_llm()
    test_single_agent()
