from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

import os
import vector_database, text_splitter, document_loader, azure_openai_api

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
    
    question = "What is T-Store?"

    answer = azure_openai_api.ask_documents(llm, question, documents)
    print("\nAsk documents: " + answer)

    chat = azure_openai_api.create_chat(llm, vectordb)

    answer = azure_openai_api.ask_vectordb(chat, question)
    print("\nAsk vectordb: " + answer['answer'])

    question = "What was my last question?"
    answer = azure_openai_api.ask_vectordb(chat, question)
    print("\nAsk vectordb: " + answer['answer'])
