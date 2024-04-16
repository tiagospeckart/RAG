# ingest_component.py

from pathlib import Path

from langchain_community.vectorstores import utils as chromautils
from langchain_community.vectorstores.chroma import Chroma

from components.document_loader import load_docs
from components.document_splitter import split_docs
from components.vector_db_manager import VectorDatabaseManager


def ingest_documents(doc_path: Path, db_path: Path) -> Chroma:
    # Load and split documents
    loaded_docs = load_docs(doc_path)

    splited_docs = split_docs(loaded_docs)

    docschroma = chromautils.filter_complex_metadata(splited_docs)

    # Load vector database
    vector_db = VectorDatabaseManager.load_or_create_vector_database(docschroma, db_path)

    return vector_db
