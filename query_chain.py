from operator import itemgetter
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_aws import BedrockLLM
from langchain_postgres import PGVector

import os
from dotenv import load_dotenv
load_dotenv()

def conversational_retrieval_chain(
        collection_name: str,
        connection_string: str,
):
    # search_kwargs
    max_num_documents = 10
    similarity_score_threshold = 0.25

    # condense_question_prompt
    # qa_prompt
    # Embedding Model
    # Condense Question Model

    # LLM to be used for reformulating question with history, if needed
    question_llm = BedrockLLM(
        model_id =  "anthropic.claude-instant-v1",
    )

    # Prompt for question reformulation with history
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            "Given the following conversation and a follow up question, "
            "rephrase the follow up question to be a standalone question, in its "
            "original language.\n\nChat History:\n{chat_history}\n"
            "Follow Up Input: {question}\nStandalone question:"
        ),
    )
    
    chat_history_stringifier = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("chat_history")]
    ) | RunnableLambda(lambda x: x.to_string())
    
    reformulate_chain = RunnableParallel({
        "question": itemgetter("question"),
        "chat_history": chat_history_stringifier,
        }) | condense_question_prompt | question_llm

    # Branch to reformulate question if there is non-empty chat history
    maybe_reformulate_with_history: RetrieverOutputLike = RunnableBranch(
        # If no chat history, then just return question
        (no_chat_history, lambda x: x["question"]),
        # Otherwise reformulate question
        reformulate_chain,
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
    )
    # Vector database for retrieval of contextual documents
    db = PGVector(
        collection_name = collection_name,
        connection = connection_string,
        embeddings = embeddings,
    )
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": max_num_documents,
            "score_threshold": similarity_score_threshold,
        },
    ).configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs",
            name="Search Parameters",
            description="Configure search parameters",
        )
    )

    # Prompt for answering question with retrieved context
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following pieces of context to answer the question "
            "at the end. If you don't know the answer, just say that you "
            "don't know, don't try to make up an answer.\n\n{context}\n\n"
            "Question: {question}\nHelpful Answer:"
        ),
    ).configurable_fields(
        template=ConfigurableField(
            id="qa_prompt",
            name="Question Prompt",
            description="Prompt for question answering with {context} and {question}",
        )
    )

    # LLM for answering with context
    model = ChatBedrock(
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    # Format the context documents into a string for use
    # in the QA chain
    format_context: RunnableParallel = RunnableParallel(
        {
            "context": (itemgetter("context") | RunnableLambda(format_docs)),
            "question": itemgetter("question"),
        }
    )
    
    chain = (
        # Required "question" string and optional "chat_history" string
        maybe_reformulate_with_history
        # Possibly-reformatted question as string
        | {
            "context": retriever,
            "question": RunnablePassthrough(),  # type: ignore[dict-item]
        }
        | {
            "context": itemgetter("context"),
            "answer": (format_context | qa_prompt | model),
        }
    )
    return chain


def no_chat_history(runnable):
    """Return True if runnable has no history that would require
    question reformulation (no key or empty history)."""
    chat_history = runnable.get("chat_history", False)
    no_history = (not chat_history) or (len(str(chat_history)) == 0)
    return no_history

        
def format_docs(docs):
    """Turn list of documents into newline-separated string"""
    context_str = "\n\n".join(doc.page_content for doc in docs)
    return context_str
