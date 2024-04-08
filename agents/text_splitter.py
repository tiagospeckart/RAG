from langchain.text_splitter import RecursiveCharacterTextSplitter

import random

def split_docs(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
  docs = text_splitter.split_documents(documents)
  
  random_result = str(random.choice(docs).page_content).replace("\n", "")
  print("\nRandom value from docs " + random_result)
  return docs
