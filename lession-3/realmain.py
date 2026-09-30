from rag import answer_question, get_sources


print("RAG Chatbot")
print("Type 'exit' to quit.")


while True:

    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        break

    try:

        response, documents = answer_question(query)

        print("\nAI:")
        print(response.content)

        print("\nSources:")

        for source in get_sources(documents):
            print("-", source)

    except Exception as e:

        print(f"\nError: {e}")