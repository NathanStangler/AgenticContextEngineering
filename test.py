from rag import RAG
from summarization import Summarization
from token_usage import count_tokens

context = [
    "I am interested in context engineering.",
    "The project is due Friday.",
    "I have a meeting at 3 PM.",
    "Class is on Tuesday and Thursday.",
    "I want to reduce the token usage.",
]

input = "What are my upcoming deadlines and meetings?"

def main():
    rag = RAG()
    summarization = Summarization()

    for c in context:
        rag.add_context(c)

    top_k = [item["text"] for item in rag.retrieve_top_k(query=input, k=3)]
    summary = summarization.summarize(top_k, max_tokens=60, min_tokens=20)

    print("Top-k Retrieved Chunks:")
    for i, chunk in enumerate(top_k, start=1):
        print(f"{i}. {chunk}")

    print("\n=Summary:")
    print(summary)

    print("\nToken Usage:")
    print(f"Retrieved context: {count_tokens(''.join(top_k))}")
    print(f"Summary: {count_tokens(summary)}")

if __name__ == "__main__":
    main()