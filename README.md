# RAG Query Chain

## Design/Implementation Questions

- Should the prompt be provided as a `str` or as a `PromptTemplate`?
  - The query chain is very brittle with respect to the `PromptTemplate`'s 
    input variables of "context" and "question". It is worth looking into
    the LangChain machinery around prompts and error checking for 
    the correct inputs behing provided.
- Should the interaction method be custom like `ask_question` or should
  the class implement the `Runnable` interface or something similar?
  - My current thought is to use the custom `ask_question` method because
    of the stateless nature of the chain and requing the `ChatMessageHistory`
    to be provided or created from scratch. It is worth looking into
    whether the retrieval chain interfaces have an affordance for passing
    in memory, in which case it might make sense to switch.
