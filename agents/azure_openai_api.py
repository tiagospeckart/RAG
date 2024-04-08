import os
from langchain_openai import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain

import openai

def create_llm():
    # Configure Azure OpenAI Service API
    openai.api_type = os.getenv('OPENAI_API_TYPE')
    openai.api_version = os.getenv('OPENAI_API_VERSION')
    openai.base_url = os.getenv('OPENAI_API_BASE')
    openai.api_key = os.getenv("OPENAI_API_KEY")

    return AzureChatOpenAI(temperature=0)

def ask(llm, question, documents):
    chain = load_qa_chain(llm, chain_type="refine")
    ## TODO use invoke()
    print("\nQuestion: " + question)
    return chain.run(input_documents=documents, question=question)
    

