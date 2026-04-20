import random
import numpy as np
from sentence_transformers import CrossEncoder

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def estimate_tokens(text: str) -> int:
    # cheap amount
    return max(1, len(text.split()))


def normalize(scores):
    scores = np.array(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    return scores


def knapsack_select(items, importance, budget):
    # Greedy approach to maximize budget use
    scored = list(zip(items, importance))

    scored.sort(key=lambda x: x[1], reverse=True)

    selected = []
    used = 0

    for text, score in scored:
        cost = estimate_tokens(text)

        if used + cost <= budget:
            selected.append(text)
            used += cost

    return selected


def generate_rag_reranker_dataset(rag, corpus, num_samples=8000, query_bank=None):
    """
    Agentic Context Engineering dataset:
    - retrieval candidates
    - importance scoring
    - token budget selection
    """

    # Load corpus into RAG
    for doc in corpus:
        rag.add_context(doc)

    encoder = _get_cross_encoder()
    dataset = []

    bank = query_bank if query_bank else QUERY_BANK

    for _ in range(num_samples):

        query = random.choice(bank)


        retrieved = rag.retrieve_top_k(query, k=15)

        if len(retrieved) < 3:
            continue

        chunks = [r["text"] for r in retrieved]


        pairs = [(query, c) for c in chunks]
        raw_scores = encoder.predict(pairs)

        importance = normalize(raw_scores)

        # Token simulation
        token_budget = random.randint(300, 1200)

        selected_context = knapsack_select(
            chunks,
            importance,
            token_budget
        )

        selected_indices = [
            i for i, c in enumerate(chunks)
            if c in selected_context
        ]


        dataset.append({
            "query": query,

            "retrieval_candidates": [
                {
                    "text": chunks[i],
                    "importance": float(importance[i])
                }
                for i in range(len(chunks))
            ],

            "selected_context": selected_indices,

            "token_budget": token_budget
        })

    return dataset