import server

context = [
    "I am interested in context engineering, so I am completing this project.",
    "The project is due Friday, but I have a lot of work to do before then.",
    "I have a meeting at 3 PM. I will leave at 2:30 PM to get there on time.",
    "Class is on Tuesday and Thursday, and there is a quiz on Thursday.",
    "I want to reduce the token usage, so I will summarize the context.",
]

user_prompt = "What are my upcoming deadlines and meetings?"

def main():
    session_id = server.create_session().get("session_id")

    for c in context:
        server.insert_response(session_id=session_id, agent_response=c)

    result = server.get_context(session_id=session_id, user_prompt=user_prompt, top_k=3, max_summary_tokens=35, min_summary_tokens=20)
    top_k_texts = [item["text"] for item in result.get("retrieved_chunks", [])]

    print("Top-k Retrieved Chunks:")
    for i, chunk in enumerate(top_k_texts, start=1):
        print(f"{i}. {chunk}")

    print("\nSummary:")
    print(result.get("context"))

    print("\nToken Usage:")
    print(f"Retrieved tokens: {result.get('retrieved_tokens')}")
    print(f"Context tokens: {result.get('context_tokens')}")

if __name__ == "__main__":
    main()