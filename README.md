# RAG Query Chain

The conversational retrieval chain demonstrated in this repository is
an example of software implementing a Generative AI conversational
experience with a a set of documents indexed for embedding-based
retrieval. It also describes the individual components of the chain,
their function, opportunities for customization, and workarounds for
issues we have encountered.

<img src="conversational_retrieval_chain.png"
     alt="Conversational Retrieval Chain diagram"
     width="800"
     height="400"/>

## Conversational Retrieval Chain Components

### Retrieval

The heart of a retrieval-augmented generation system is the retriever
that takes the input query and returns a set of texts related to the
query that will be used to answer the question.

```
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
    )
    # Vector database for retrieval of contextual documents
    db = PGVector(
        collection_name=collection_name,
        connection=connection_string,
        embeddings=embeddings,
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
```

Some of the settings that can be used to affect retrieval are:

- The maximum number of documents to return (`k`)
- A filter condition on document metadata which is specific to your
  documents' metadata and to the filter language for your vector
  database (`filter`)
- A similarity score threshold. This is available in the retrievers
  created from the `PGvector` vector database and only returns
  documents that score above a specified relevance threshold
  (`search_type` and `score_threshold`)


### Memory

The memory in a conversational retrieval chain keeps track of the back
and forth interactions between the user and the chain to be used
provide context, often in the form a rephrasing a later question
in the conversation chain to be context-free using the earlier
question/answer pairs, although this can be done in other ways
and the question/answer memory can be used for other purposes as well.

In this implementation the memory is maintained outside of the
conversational-retrieval chain and is passed to the chain using the
`chat_history` key.
    
```
    # LLM to be used for reformulating question with history, if needed
    question_llm = BedrockLLM(
        model_id="anthropic.claude-instant-v1",
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

    reformulate_chain = (
        RunnableParallel(
            {
                "question": itemgetter("question"),
                "chat_history": chat_history_stringifier,
            }
        )
        | condense_question_prompt
        | question_llm
    )

    # Branch to reformulate question if there is non-empty chat history
    maybe_reformulate_with_history: RetrieverOutputLike = RunnableBranch(
        # If no chat history, then just return question
        (no_chat_history, lambda x: x["question"]),
        # Otherwise reformulate question
        reformulate_chain,
    )

```

If there is chat history this part of the chain runs the question
and the chat history through the `condense_question_prompt` to
make the question context free.

### Response Generation

The full chain takes the context from the retrieval step and passes it
with the (possibly reformulated) question  to the `qa_prompt` to
produce a response. The retrieved context is also passed as output in
case it is needed for analysis or display

```
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following pieces of context to answer the question "
            "at the end. If you don't know the answer, just say that you "
            "don't know, don't try to make up an answer. Skip the preamble "
            "and only provide the answer.\n\n{context}\n\n"
            "Question: {question}\nHelpful Answer:"
        ),
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
```

## Files

- [`query_chain.py`](query_chain.py) - The basic RAG conversational retrieval chain.
- [`query_chain_test.py`](query_chain_test.py) - A test script runs the query
  chain in non-streaming and streaming modes.
- [`query_chain_ui.py`](query_chain_ui.py) - A
  [streamlit](https://streamlit.io/) UI for simple chatbot interaction
  with a query chain.
- [`qc_gradio_ui.py`](qc_gradio_ui.py) - A [Gradio](https://www.gradio.app/) UI demonstrating the
  same type of chatbot.

## To Do

- [x] Add labeled diagram of flow
- [x] Add `requirements.txt`
- [ ] Add installation instructions
- [x] Fix streaming in streamlit UI
- [x] Modify README.md to match new LCEL code
- [ ] Add basic embedding generation
- [x] Make a Gradio UI.
