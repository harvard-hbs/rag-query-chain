from bedrock_postgres_chain import BedrockPostgresChain

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
    response = query_chain.ask_question("Is language a social construct?")
    print(response["answer"])

    print("With streaming...")
    query_chain = BedrockPostgresChain(
        model_id=LLM_MODEL_ID,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        search_kwargs={"k": MAX_RETRIEVAL_COUNT},
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    response = query_chain.ask_question("Is language a social construct?")
    print()
    print("Done.")
    
        
if __name__ == "__main__":
    main()
       
