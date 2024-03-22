from query_chain import QueryChain

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.chat_models import BedrockChat

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
            response_if_no_docs_found: str = None,
            callbacks=None,
    ):
        if callbacks is not None:
            streaming = True
        else:
            streaming = False
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
        )
        db = PGVector(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_string=connection_string,
        )
        retriever = db.as_retriever(
            search_kwargs=search_kwargs,
        )
        model = BedrockChat(
            model_id=model_id,
            streaming=streaming,
        )
        super().__init__(
            model=model,
            retriever=retriever,
            prompt_template=prompt_template,
            response_if_no_docs_found=response_if_no_docs_found,
            callbacks=callbacks,
        )

