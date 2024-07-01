import os

import json
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
import streamlit as st

from query_chain import conversational_retrieval_chain

from dotenv import load_dotenv
load_dotenv()


def create_chat_history():
    history = InMemoryChatMessageHistory()
    history.add_ai_message("How can I help you?")
    return history

def initialize_query_chain():
    max_num_docs = st.session_state["max_num_docs"]
    score_threshold = st.session_state["similarity_cutoff"]
    chain = conversational_retrieval_chain(
        collection_name=os.getenv("COLLECTION_NAME"),
        connection_string=os.getenv("CONNECTION_STRING"),
    ).with_config(
        configurable={
            "search_kwargs": {
                "k": max_num_docs,
                "score_threshold": score_threshold,
            }
        }
    )
    st.session_state["query_chain"] = chain

st.sidebar.number_input(
    label="Maximum number of documents",
    min_value=1,
    max_value=30,
    value=10,
    key="max_num_docs",
    on_change=initialize_query_chain,
)

st.sidebar.slider(
    label="Similarity cutoff",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    key="similarity_cutoff",
    on_change=initialize_query_chain,
)

def stream_response(response_generator):
    for resp_part in response_generator:
        if "answer" in resp_part:
            yield resp_part["answer"]
        elif "context" in resp_part:
            st.session_state["context"] = resp_part["context"]
        else:
            print("Warning: unknown response type: {resp_part}")

def stream_model_query(question):
    # Reset session context state in case message has no context
    if "context" in st.session_state:
        del st.session_state["context"]
    query_chain = st.session_state["query_chain"]
    query_params = {"question": question}
    chat_history = st.session_state["chat_history"]
    if len(chat_history.messages) > 1:
        # Don't pass history if first question
        query_params["chat_history"] = chat_history.messages
    response_generator = stream_response(
        query_chain.stream(query_params)
    )
    return response_generator

def write_ref_doc(doc):
    # Don't include "text" field if it exists in the metadata
    md = doc.metadata
    md_no_key = {key: md[key] for key in md if key != "text"}
    st.markdown(f"- {md_no_key}")

def write_context(documents):
    st.markdown(f"### References ({len(documents)})")
    for doc in documents:
        write_ref_doc(doc)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = create_chat_history()

if "query_chain" not in st.session_state:
    initialize_query_chain()
    
st.markdown("## Chatbot")

for message in st.session_state["chat_history"].messages:
    with st.chat_message(message.type):
        st.markdown(message.content)
        if message.response_metadata:
            context = message.response_metadata["context"]
            write_context(context)
        
if question := st.chat_input("Your message"):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("ai"):
        # Stream `answer` part of response and save `context`.
        response = st.write_stream(stream_model_query(question))
        # Write a representation of `context` if present
        if "context" in st.session_state:
            write_context(st.session_state["context"])

    chat_history = st.session_state["chat_history"]
    chat_history.add_user_message(question)
    if "context" in st.session_state:
        # If there are context documents, save them as part of
        # the message so they can be shown in the history.
        context = st.session_state["context"]
        ai_message = AIMessage(
            response,
            response_metadata={"context": context}
        )
        chat_history.add_message(ai_message)
    else:
        chat_history.add_ai_message(response)
    
