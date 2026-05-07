from context_manager import ContextManager

class ContextEngine:
    def __init__(self):
        self.manager = ContextManager()

    def ingest_turn(self, role, text, metadata=None):
        return self.manager.ingest_turn(role=role, text=text, metadata=metadata)

    def build_prompt_context(self, query, token_budget, mode="dynamic_context", top_k=5, use_reranker=True):
        return self.manager.build_prompt_context(
            query=query,
            token_budget=token_budget,
            mode=mode,
            top_k=top_k,
            use_reranker=use_reranker,
        )

    def record_response(self, response_text, metadata=None):
        return self.manager.record_response(response_text=response_text, metadata=metadata)

    def get_metrics(self):
        return self.manager.get_metrics()

def main():
    engine = ContextEngine()
    engine.ingest_turn("user", "I care about reducing token costs.")
    engine.ingest_turn("assistant", "We can use retrieval and summary compression.")
    engine.ingest_turn("user", "My final report is due Friday and I have a 3 PM meeting.")

    pkg = engine.build_prompt_context(
        query="What are my priorities and deadlines?",
        token_budget=120,
        mode="dynamic_context",
        top_k=4,
    )

    print("Summary:")
    print(pkg["summary"])

    print("\nRetrieved Chunks:")
    for idx, chunk in enumerate(pkg.get("reranked_chunks", []), start=1):
        print(f"{idx}. {chunk.get('text', '')}")

    print("\nContext package:")
    print(pkg["context"])
    print("\nMetrics:")
    print(engine.get_metrics())

if __name__ == "__main__":
    main()