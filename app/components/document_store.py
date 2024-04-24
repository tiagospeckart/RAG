import logging

from injector import singleton
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import constants

logger = logging.getLogger(__name__)
    
@singleton
class FAISSDocumentStore:
    """
    FAISSDocumentStore represents a singleton document store utilizing the FAISS vectorization technique
    for efficient text representation and retrieval.

    This class enables chunking and vectorization of documents for advanced text-based operations.

    Parameters:
        chunk_size (int): Size of each document chunk in characters for processing.
        chunk_overlap (int): Number of overlapping characters between consecutive document chunks.

    Attributes:
        chunk_size (int): Size of each document chunk.
        chunk_overlap (int): Overlapping characters between chunks.
        embedding_function (SentenceTransformerEmbeddings): Sentence embedding model for text vectorization.
        faiss_db (FAISS): FAISS vector database initialized with chunked and vectorized documents.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Initialize the FAISSDocumentStore with specified chunking parameters and setup the FAISS vector database.

        Args:
            chunk_size (int): Size of each document chunk in characters for processing.
            chunk_overlap (int): Number of overlapping characters between consecutive document chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.faiss_db = self._initialize_faiss_db()

    def _initialize_faiss_db(self) -> FAISS:
        """
        Initialize the FAISS vector database by loading and processing documents from a directory.

        Returns:
            FAISS: Initialized FAISS vector database.
        """
        logger.debug("Initializing FAISS Document store component=%s", type(self).__name__)
        
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

        # Create FAISS vector database injesting the Chunks with the chosen Embedding Function
        return FAISS.from_documents(chunks, self.embedding_function)
    