from query_chain import QueryChain

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_community.llms.bedrock import Bedrock
from langchain_community.vectorstores.pgvector import PGVector

import os
from dotenv import load_dotenv
load_dotenv()

class BedrockPostgresChain(QueryChain):
    def __init__(
            self,
            model_id: str,
            collection_name: str,
            connection_string: str,
            search_kwargs = {},
            prompt_template: str = None,
            search_type: str = "similarity",
            response_if_no_docs_found: str = None,
            streaming=False,
            callbacks=None,
    ):
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
        )
        db = PGVector(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_string=connection_string,
        )
        retriever = db.as_retriever(
            search_type = search_type,
            search_kwargs = search_kwargs,
        )
        model = BedrockChat(
            model_id=model_id,
            streaming=streaming,
            callbacks=callbacks,
        )
        question_llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            streaming=False,
        )
        super().__init__(
            model=model,
            retriever=retriever,
            prompt_template=prompt_template,
            condense_question_llm=question_llm,
            response_if_no_docs_found=response_if_no_docs_found,
        )

