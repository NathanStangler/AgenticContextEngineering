import argparse
import json
import math
from collections import defaultdict
import pandas as pd
from context_manager import ContextManager

def dcg_at_k(relevances, k):
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        rel = relevances[i]
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(relevances, k):
    if not relevances:
        return 0.0
    ideal = sorted(relevances, reverse=True)
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(relevances, k, threshold):
    if not relevances:
        return 0.0
    relevant = [1 if r >= threshold else 0 for r in relevances]
    total_relevant = sum(relevant)
    if total_relevant == 0:
        return 0.0
    hits = sum(relevant[:k])
    return hits / total_relevant

def normalize_text(text):
    return " ".join(text.lower().split())

def relevance_for_chunk(chunk, candidates):
    if not chunk:
        return 0.0
    chunk_norm = normalize_text(chunk)
    best = 0.0
    for cand_text, importance in candidates:
        if cand_text in chunk_norm or chunk_norm in cand_text:
            if importance > best:
                best = importance
    return best

def build_query_and_history(conversation):
    if not conversation:
        return None, []

    last_user_idx = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return None, []

    query = conversation[last_user_idx].get("content", "").strip()
    history = conversation[:last_user_idx]
    return query, history

def evaluate_sample(sample, manager, modes, top_k, token_budget, threshold, use_reranker):
    conversation = sample.get("conversation", [])
    candidates = sample.get("retrieval_candidates", [])

    query, history = build_query_and_history(conversation)
    if not query:
        return None

    manager.session.turns = []
    manager.session.rag.reset()
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if content:
            manager.ingest_turn(role, content)

    candidate_importance = [
        (normalize_text(c["text"]), float(c.get("importance", 0.0)))
        for c in candidates
        if c.get("text")
    ]

    results = {}
    for mode in modes:
        rerank = use_reranker and mode.endswith("_rerank")
        base_mode = mode.replace("_rerank", "")
        pkg = manager.build_prompt_context(
            query=query,
            token_budget=token_budget,
            mode=base_mode,
            top_k=top_k,
            use_reranker=rerank,
        )

        ranked = pkg.get("reranked_chunks") if rerank else pkg.get("retrieved_chunks")
        ranked_texts = [item.get("text", "") for item in ranked]
        relevances = [relevance_for_chunk(text, candidate_importance) for text in ranked_texts]

        results[mode] = {
            "context_tokens": pkg.get("context_tokens", 0),
            "ndcg": ndcg_at_k(relevances, top_k),
            "recall": recall_at_k(relevances, top_k, threshold),
        }

    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark context modes and reranking.")
    parser.add_argument("--data", default="data/processed/querycontext_dataset.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--reranker-weights", default="train/model.pth")
    parser.add_argument("--modes", nargs="+", default=["rag", "dynamic_context", "rag_rerank", "dynamic_context_rerank"])
    args = parser.parse_args()

    with open(args.data, "r") as f:
        dataset = json.load(f)

    manager = ContextManager(reranker_weights=args.reranker_weights)
    totals = defaultdict(lambda: {"ndcg": 0.0, "recall": 0.0, "context_tokens": 0.0, "count": 0})

    rows = []
    per_k = list(range(1, 6))

    for k in per_k:
        totals = defaultdict(lambda: {"ndcg": 0.0, "recall": 0.0, "context_tokens": 0.0, "count": 0})

        for sample in dataset[: args.limit]:
            result = evaluate_sample(
                sample,
                manager,
                args.modes,
                k,
                args.token_budget,
                args.threshold,
                use_reranker=True,
            )
            if result is None:
                continue
            for mode, metrics in result.items():
                totals[mode]["ndcg"] += metrics["ndcg"]
                totals[mode]["recall"] += metrics["recall"]
                totals[mode]["context_tokens"] += metrics["context_tokens"]
                totals[mode]["count"] += 1

        for mode in args.modes:
            count = totals[mode]["count"]
            if count == 0:
                continue
            rows.append(
                {
                    "mode": mode,
                    "k": k,
                    "ndcg": totals[mode]["ndcg"] / count,
                    "recall": totals[mode]["recall"] / count,
                    "context_tokens": totals[mode]["context_tokens"] / count,
                    "samples": count,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["mode", "k"]).reset_index(drop=True)
    print("\nBenchmark Results")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()