from bedrock_postgres_chain import BedrockPostgresChain

from langchain.memory import ChatMessageHistory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

MAX_RETRIEVAL_COUNT=10
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")

def main():
    print("Without streaming...")
    query_chain = BedrockPostgresChain(
        model_id=LLM_MODEL_ID,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        search_kwargs={"k": MAX_RETRIEVAL_COUNT},
    )
    chat_history = ChatMessageHistory()
    query = "Is language a social construct?"
    response = query_chain.ask_question(query, chat_history)
    print(chat_history)
    query = "What parts are not social?",
    response = query_chain.ask_question(query, chat_history)
    print(chat_history)
    print()

    print("With streaming...")
    query_chain = BedrockPostgresChain(
        model_id=LLM_MODEL_ID,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        search_kwargs={"k": MAX_RETRIEVAL_COUNT},
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    chat_history = ChatMessageHistory()
    query = "Is language a social construct?"
    print(query)
    response = query_chain.ask_question(query, chat_history)
    print("\n")
    query = "What parts are not social?"
    print(query)
    response = query_chain.ask_question(query, chat_history)
    print()
    print("Done.")
    
        
if __name__ == "__main__":
    main()
       
