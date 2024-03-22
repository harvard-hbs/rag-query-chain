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

def new_chat_history():
    logger.info("Creating new chat history")
    chat_history = ChatMessageHistory()
    chat_history.add_ai_message("How can I help you?")
    return chat_history

def create_session():
    session_id = str(uuid.uuid4())
    logger.info(f"Creating new session: {session_id}")
    st.session_state["chat_history"] = new_chat_history()
    st.session_state["query_chain"] = BedrockPostgresChain(
        model_id = LLM_MODEL_ID,
        collection_name = COLLECTION_NAME,
        connection_string = CONNECTION_STRING,
        search_kwargs = {"k": MAX_RETRIEVAL_COUNT},
    )
    logger.info(f"Done with session creation: {session_id}")
    return session_id

def process_query():
    chat_query = st.session_state["chat_query"]
    session_id = st.session_state["session_id"]
    logger.info(f"Processing query ({session_id}): '{chat_query}'")
    chat_history = st.session_state["chat_history"]
    query_chain = st.session_state["query_chain"]
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
    
st.chat_input("Your message", key="chat_query", on_submit=process_query)
