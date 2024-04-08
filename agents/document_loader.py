import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_docs():  
  directory = os.getenv("DOCUMENTS_FOLDER")
  
  loader = DirectoryLoader(directory, glob="*.md", loader_cls=TextLoader)
  documents = loader.load()
  print("\nNumber of documents loaded: " + str(len(documents)))
  
  return documents
