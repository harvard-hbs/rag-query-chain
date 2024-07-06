from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
import gradio as gr
import os

from query_chain import conversational_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

query_chain = conversational_retrieval_chain(
    collection_name=os.getenv("COLLECTION_NAME"),
    connection_string=os.getenv("CONNECTION_STRING"),
)

def format_chat_history(history):
    """Format the message history from the `Chatbot`
    component (a list of string pairs) for LangChain."""
    chat_history = InMemoryChatMessageHistory()
    for human_msg, ai_msg in history:
        chat_history.add_user_message(human_msg)
        chat_history.add_ai_message(ai_msg)
    return chat_history

def query_model(message, history):
    chat_history = format_chat_history(history)
    response_generator = query_chain.stream({
        "question": message,
        "chat_history": chat_history.messages,
    })
    partial_response = ""
    context = []
    for resp_part in response_generator:
        if "context" in resp_part:
            # Keep track of RAG context for later streaming
            context = resp_part["context"]
        elif "answer" in resp_part:
            "Build and yield reasponse as chunks are streamed."
            partial_response = (
                partial_response + resp_part["answer"].content
            )
            yield partial_response
    # Stream RAG context saved above
    for doc in context:
        label = doc.metadata["label"]
        partial_response = partial_response + f"\n- {label}"
        yield partial_response

gr.ChatInterface(query_model).launch()
