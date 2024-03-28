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

class CallbackTester(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print("LLM start...")

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        print("Chat Model start...")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print("LLM end...")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        print("Chain start....")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        print("Chain end...")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        print(f"Text {text}...")

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
        # callbacks=[StreamingStdOutCallbackHandler()],
        callbacks=[CallbackTester()],
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
       
