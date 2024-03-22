import os
import uuid

import streamlit as st
from streamlit.logger import get_logger
from dotenv import load_dotenv
from bedrock_postgres_chain import BedrockPostgresChain
from langchain.memory import ChatMessageHistory

logger = get_logger(__name__)
load_dotenv()

MAX_RETRIEVAL_COUNT=10
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")

def create_session():
    session_id = str(uuid.uuid64())
    st.session_state["chat_history"] = ChatMessageHistory()
    st.session_state["query_chain"] = BedrockPostgresChain(
        model_id = LLM_MODEL_ID,
        collection_name = COLLECTION_NAME,
        connection_string = CONNECTION_STRING,
        search_kwargs = {"k": MAX_RETRIEVAL_COUNT},
    )
    return session_id

if "session_id" not in st.session_state:
    st.session_state["session_id"] = create_session()

st.title("Document Chatbot")

st.write(
    """This conversational interface allows you to interact with
indexed content."""
)
    
