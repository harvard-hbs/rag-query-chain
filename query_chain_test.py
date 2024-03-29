from bedrock_postgres_chain import BedrockPostgresChain

from langchain.memory import ChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from dotenv import load_dotenv
load_dotenv()

MAX_RETRIEVAL_COUNT=10
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")

def print_source_documents(response):
    for sd in response["source_documents"]:
        md = sd.metadata
        print(f"    {md['chapter_id']} - {md['text'][:60]}")
    
def main():
    print("Without streaming...")
    query_chain = BedrockPostgresChain(
        model_id = LLM_MODEL_ID,
        collection_name = COLLECTION_NAME,
        connection_string = CONNECTION_STRING,
        search_type = "similarity_score_threshold",
        search_kwargs = {
            "k": MAX_RETRIEVAL_COUNT,
            "score_threshold": 0.5,
        },
    )
    chat_history = ChatMessageHistory()
    query = "Is language a social construct?"
    print(f"Question: {query}")
    response = query_chain.ask_question(query, chat_history)
    print(f"Answer: {response['answer'][:60]}")
    print_source_documents(response)

    print("With streaming...")
    query_chain = BedrockPostgresChain(
        model_id=LLM_MODEL_ID,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        search_kwargs={"k": MAX_RETRIEVAL_COUNT},
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    query = "What parts are not social?",
    print(f"Question: {query}")
    response = query_chain.ask_question(query, chat_history)
    print("\n")
    print_source_documents(response)

    query = "Who won the world series in 2023?"
    print(f"Question: {query}")
    response = query_chain.ask_question(query, chat_history)
    print("\n")
    print_source_documents(response)
    
    print("Done.")
    
        
if __name__ == "__main__":
    main()
       
