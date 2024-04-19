import logging

from injector import singleton
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import constants

logger = logging.getLogger(__name__)

@singleton
class ChromaDocumentStore:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_db = self._initialize_chroma_db()

    def _initialize_chroma_db(self) -> Chroma:
        documents_path = constants.DOCUMENTS_PATH
        loader = DirectoryLoader(documents_path, glob="**/*.md")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_documents = text_splitter.split_documents(documents)

        chroma_db = Chroma.from_documents(split_documents, self.embedding_function)
        return chroma_db

    def query_similar_documents(self, query: str):
        return self.chroma_db.similarity_search(query)

    def add_documents(self, new_documents):
        self.chroma_db.add_documents(new_documents)

    def update_document(self, document_id: str, updated_document):
        self.chroma_db.update_document(document_id, updated_document)

