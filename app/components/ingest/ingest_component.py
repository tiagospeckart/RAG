from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path


# Markdown loader
def load_docs(directory: Path) -> list[Document]:
    # Ensure the provided directory exists
    if not directory.is_dir():
        raise ValueError(f"Directory '{directory}' does not exist or is not a valid directory.")

    # Initialize an empty list to hold loaded documents
    loaded_documents: list[Document] = []

    # Iterate over each file in the directory
    for file_path in directory.iterdir():
        # Check if the item is a file
        if file_path.is_file():
            # Load the document using UnstructuredMarkdownLoader
            md_loader = UnstructuredMarkdownLoader(file_path, mode="elements")
            loaded_documents.extend(md_loader.load())

    return loaded_documents


def split_docs(documents_list,
               chunk_size=1000,
               chunk_overlap=20) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = text_splitter.split_documents(documents_list)
    return chunks

