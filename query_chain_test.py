from query_chain import conversational_retrieval_chain

from langchain_core.chat_history import InMemoryChatMessageHistory

import os
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()

    
def main():
    chat_history = InMemoryChatMessageHistory()
    chain = conversational_retrieval_chain(
        collection_name = os.getenv("COLLECTION_NAME"),
        connection_string = os.getenv("CONNECTION_STRING"),
    )

    print("Without streaming...")
    question = "Who is Aragorn?"
    print(f"{question=}")
    response = chain.invoke({"question": question})
    print(response["answer"].content)
    print("References:")
    for doc in response["context"]:
        pprint(doc.metadata)
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response["answer"])

    print("With streaming...")
    question = "How did he meet Elrond?"
    for resp_part in chain.stream({
            "question": question,
            "chat_history": chat_history.messages,
    }):
        if "context" in resp_part:
            print("References:")
            for doc in resp_part["context"]:
                pprint(doc.metadata)
        else:
            assert("answer" in resp_part)
            print(resp_part["answer"].content, end="")
    print()
    print("Done.")
    
        
if __name__ == "__main__":
    main()
       
