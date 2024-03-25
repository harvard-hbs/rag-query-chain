# RAG Query Chain

## Conversational Retrieval Chain Components

### Retrieval

The heart of a retrieval-augmented generation system is the retriever
that takes the input query and returns a set of texts related to the
query that will be used to answer the question.

- An embedding model (e.g., `amazon.titan-embed-text-v1` in AWS Bedrock)
- A vector database (e.g., Postgres PGVector)
- LangChain objects for vector database and retriever (e.g., `PGVector.as_retriever()`)

Some of the settings that can be used to affect retrieval are (some of these are
specific to `PGVector`):

- The maximum number of documents to return (`k`)
- A filter condition on document metadata (`filter`)
- A similarity score threshold (`search_type` and `score_threshold`)

Here as an example of creating a retriever with a maximum number
of returned documents of 10, a filter to only return PDF documents
(using some made-up document metadata) and with a similarity
score threshold of 0.5.

```
retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k": 10,
        "filter": {"doc_type": "pdf"},
        "score_threshold": 0.5,
    }
)
```

### Memory



### Conversational Context
### Response Generation

## Files

- [`query_chain.py`](query_chain.py) - The basic RAG conversational retrieval chain.
- [`query_chain_test.py`](query_chain_test.py) - A test script that instantiates
  a `QueryChain` using `BedrockChat` and `PGVector` and makes a query, with and
  without streaming.
- [`query_chain_ui.py`](query_chain_ui.py) - A streamlit UI for simple chatbot
  interaction with a query chain.
- [`bedrock_postgres_chain.py`](bedrock_postgres_chain.py) - A subclass of
  `QueryChain` specific to `BedrockChat` and `PGVector`.

## Guiding Principles

- Create a working skeleton of the conversational retrieval
  chain where the basic functions are maintained in one class
  that can be used in many contexts.
- The inputs of the base `QueryChain` class are the large
  language model (LLM), retriever, and system prompt at
  creation time, and then the new question/query and the
  the chat message history at query time.
- There is a `BedrockPostgresChain` subclass that is specialized
  to `BedrockChat` and a `PGVector`-based retriever.
- Streaming is enabled by using an LLM is streaming mode and
  by providing one or more instances of `BaseCallbackHandler`.

## Design/Implementation Questions

- Should the prompt be provided as a `str` or as a `PromptTemplate`?
  - The query chain is very brittle with respect to the `PromptTemplate`'s 
    input variables of `"context"` and `"question"`. It is worth looking into
    the LangChain machinery around prompts to see if we can enforce the
    presence of the correct inputs and use that to help enforce.
- Should the interaction method be custom like `ask_question` or should
  the class implement the `Runnable` interface or something similar?
  - My current thought is to use the custom `ask_question` method because
    of the stateless nature of the chain and requing the `ChatMessageHistory`
    to be provided or created from scratch. It is worth looking into
    whether the retrieval chain interfaces have an affordance for passing
    in memory, in which case it might make sense to switch.
