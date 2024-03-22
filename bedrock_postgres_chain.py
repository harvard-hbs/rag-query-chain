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
            prompt_template: str = None,
            response_if_no_docs_found: str = None,
            callbacks=None,
    ):
        pass
