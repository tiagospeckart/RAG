import os

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def generate_embeddings():
    return SentenceTransformerEmbeddings(model_name=os.getenv("EMBEDDINGS_MODEL"))


def create_vector_database(splited_docs, persist_directory):
    vectordb = Chroma.from_documents(splited_docs, generate_embeddings(),persist_directory=persist_directory)
    
    vectordb.persist()

    ## Test the vector database
    query = "Supplier"
    matching_docs = vectordb.similarity_search(query, 1)
    print("\nMatching docs for: " + query + "\n" + str(matching_docs[0]))

    return vectordb

