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

def query_model(message, history):
    chat_history = InMemoryChatMessageHistory()
    for human_msg, ai_msg in history:
        chat_history.add_user_message(human_msg)
        chat_history.add_ai_message(ai_msg)
    partial_response = ""
    response_generator = query_chain.stream({
        "question": message,
        "chat_history": chat_history.messages,
    })
    for resp_part in response_generator:
        if "answer" in resp_part:
            partial_response = (
                partial_response + resp_part["answer"].content
            )
        yield partial_response

gr.ChatInterface(query_model).launch()
