from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(documents_list,
               chunk_size=1000,
               chunk_overlap=20) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = text_splitter.split_documents(documents_list)
    return chunks
