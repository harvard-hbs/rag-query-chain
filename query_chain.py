from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory

class QueryChain:
    def __init__(
            self,
            model: BaseLanguageModel,
            retriever: BaseRetriever,
            prompt_template: str = None,
            response_if_no_docs_found: str = None,
            callbacks=None,
    ):
        self.model = model
        if prompt_template:
            qa_prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"],
            )
            self.combine_docs_chain_kwargs = {"prompt": qa_prompt}
        else:
            self.combine_docs_chain_kwargs = None
        self.retriever = retriever
        self.response_if_no_docs_found = response_if_no_docs_found
        self.callbacks = callbacks

    def ask_question(
            self,
            question: str,
            chat_history: ChatMessageHistory = None,
            metadata = {},
    ):
        if chat_history is None:
            chat_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        query_chain = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            memory=memory,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=False,
            response_if_no_docs_found=self.response_if_no_docs_found,
            combine_docs_chain_kwargs=self.combine_docs_chain_kwargs,
        )
        response = query_chain.invoke(
            {"question": question},
            config={"callbacks": self.callbacks},
        )
        return response
