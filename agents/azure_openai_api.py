import os
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain.memory import ConversationBufferMemory

import openai

def create_llm():
    # Configure Azure OpenAI Service API
    openai.api_type = os.getenv('OPENAI_API_TYPE')
    openai.api_version = os.getenv('OPENAI_API_VERSION')
    openai.base_url = os.getenv('OPENAI_API_BASE')
    openai.api_key = os.getenv("OPENAI_API_KEY")

    return AzureChatOpenAI(temperature=0)

def ask_documents(llm, question, documents):
    chain = load_qa_chain(llm, chain_type="refine")
    ## TODO use invoke()
    print("\nQuestion: " + question)
    return chain.run(input_documents=documents, question=question)
    

def ask_vectordb(chat, question):
    print("\nQuestion: " + question)
    return chat.invoke(
        {"question": question}
    )


def create_chat(llm, vectordb):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=False,
        memory=ConversationBufferMemory(memory_key="chat_history", output_key="answer", input_key="question")
    )