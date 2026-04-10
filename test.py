import server

context = [
    "I am interested in context engineering.",
    "The project is due Friday.",
    "I have a meeting at 3 PM.",
    "Class is on Tuesday and Thursday.",
    "I want to reduce the token usage.",
]

user_prompt = "What are my upcoming deadlines and meetings?"

def main():
    session_id = server.create_session().get("session_id")

    for c in context:
        server.insert_response(session_id=session_id, agent_response=c)

    result = server.get_context(session_id=session_id, user_prompt=user_prompt, top_k=3, max_summary_tokens=60, min_summary_tokens=20)
    top_k_texts = [item["text"] for item in result.get("retrieved_chunks", [])]

    print("Top-k Retrieved Chunks:")
    for i, chunk in enumerate(top_k_texts, start=1):
        print(f"{i}. {chunk}")

    print("\n=Summary:")
    print(result.get("context"))

    print("\nToken Usage:")
    print(f"Retrieved tokens: {result.get('retrieved_tokens')}")
    print(f"Context tokens: {result.get('context_tokens')}")

if __name__ == "__main__":
    main()