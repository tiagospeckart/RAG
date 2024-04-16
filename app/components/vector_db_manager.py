import os

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from settings.settings import settings


def generate_embeddings():
    return SentenceTransformerEmbeddings(model_name=settings().azopenai.embedding_model)


class VectorDatabaseManager:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory

    def create_vector_database(self, splited_docs) -> Chroma:
        embedding_function = generate_embeddings()
        vectordb = Chroma.from_documents(
            splited_docs,
            embedding_function,
            persist_directory=self.persist_directory
        )

        vectordb.persist()

        return vectordb

    @classmethod
    def load_or_create_vector_database(cls, splited_docs, persist_directory) -> Chroma:
        if os.path.exists(persist_directory):
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=generate_embeddings()
            )
        else:
            return cls(persist_directory).create_vector_database(splited_docs)

