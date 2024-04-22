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
    """
    ChromaDocumentStore represents a singleton document store utilizing the Chroma vectorization technique
    for efficient text representation and retrieval.

    This class enables chunking and vectorization of documents for advanced text-based operations.

    Parameters:
        chunk_size (int): Size of each document chunk in characters for processing.
        chunk_overlap (int): Number of overlapping characters between consecutive document chunks.

    Attributes:
        chunk_size (int): Size of each document chunk.
        chunk_overlap (int): Overlapping characters between chunks.
        embedding_function (SentenceTransformerEmbeddings): Sentence embedding model for text vectorization.
        chroma_db (Chroma): Chroma vector database initialized with chunked and vectorized documents.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Initialize the ChromaDocumentStore with specified chunking parameters and setup the Chroma vector database.

        Args:
            chunk_size (int): Size of each document chunk in characters for processing.
            chunk_overlap (int): Number of overlapping characters between consecutive document chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_db = self._initialize_chroma_db()

    def _initialize_chroma_db(self) -> Chroma:
        """
        Initialize the Chroma vector database by loading and processing documents from a directory.

        Returns:
            Chroma: Initialized Chroma vector database.
        """
        logger.debug("Initializing Chroma Document store component=%s", type(self).__name__)
        
        # Load all Markdown files in the Documents Path as a List of Documents
        documents_path = constants.DOCUMENTS_PATH
        text_loader_kwargs = {'autodetect_encoding': True}
        loader = DirectoryLoader(documents_path, glob="**/*.md", loader_kwargs=text_loader_kwargs)
        documents = loader.load()

        # Split all Documents according to Chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        print("Doc Split Size : " + str(len(chunks)))
        print(chunks[0])

        # Create Chroma vector database injesting the Chunks with the chosen Embedding Function
        return Chroma.from_documents(chunks, self.embedding_function)
    