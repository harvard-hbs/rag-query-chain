from query_chain import QueryChain

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.chat_models import BedrockChat
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

STREAMING=True
MAX_RETRIEVAL_COUNT=10

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")

def main():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
    )
    db = PGVector(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    retriever = db.as_retriever(search_kwargs={"k": MAX_RETRIEVAL_COUNT})
    model = BedrockChat(
        model_id=LLM_MODEL_ID,
        streaming=STREAMING,
    )
    if STREAMING:
        callbacks = [StreamingStdOutCallbackHandler()]
    else:
        callbacks = None
    query_chain = QueryChain(
        model=model,
        retriever=retriever,
        callbacks=callbacks,
    )
    response = query_chain.ask_question("Is language a social construct?")
    if not STREAMING:
        print(response["answer"])

if __name__ == "__main__":
    main()
       
