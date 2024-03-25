import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import streamlit as st
from streamlit.logger import get_logger

from dotenv import load_dotenv
from bedrock_postgres_chain import BedrockPostgresChain
from langchain.memory import ChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

logger = get_logger(__name__)
load_dotenv()

MAX_RETRIEVAL_COUNT=10
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")

class StreamingHandler(BaseCallbackHandler):
    def __init__(self):
        logger.info("StreamingHandler __init__")
        
    def on_chat_model_start(
            self, serialized, messages, **kwargs
    ) -> Any:
        """Run when Chat Model starts running."""
        logger.info("Chat Model start")
        self.text = st.empty()
        self.answer = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.answer += token
        self.text.write(self.answer)

def new_chat_history():
    logger.info("Creating new chat history")
    chat_history = ChatMessageHistory()
    chat_history.add_ai_message("How can I help you?")
    return chat_history

def create_session():
    session_id = str(uuid.uuid4())
    logger.info(f"Creating new session: {session_id}")
    st.session_state["chat_history"] = new_chat_history()
    logger.info(f"Done with session creation: {session_id}")
    return session_id

def query_callback():
    chat_query = st.session_state["chat_query"]
    session_id = st.session_state["session_id"]
    chat_history = st.session_state["chat_history"]
    process_query(
        chat_query,
        session_id,
        chat_history,
    )

def process_query(
        chat_query,
        session_id,
        chat_history,
):
    logger.info(f"Processing query ({session_id}): '{chat_query}'")
    with st.chat_message("human"):
        st.write(chat_query)
    with st.chat_message("ai"):
        query_chain = BedrockPostgresChain(
            model_id = LLM_MODEL_ID,
            collection_name = COLLECTION_NAME,
            connection_string = CONNECTION_STRING,
            search_kwargs = {"k": MAX_RETRIEVAL_COUNT},
            streaming = True,
            callbacks=[StreamingHandler()],
    )
    response = query_chain.ask_question(chat_query, chat_history)
    logger.info(f"Query response ({session_id}): {response['answer']}")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = create_session()

st.title("Document Chatbot")

st.write(
    """This conversational interface allows you to interact with
indexed content."""
)

for msg in st.session_state["chat_history"].messages:
    with st.chat_message(msg.type):
        st.write(msg.content)
    
st.chat_input("Your message", key="chat_query", on_submit=query_callback)
