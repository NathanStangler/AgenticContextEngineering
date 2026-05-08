import argparse
import json
import os
from collections import defaultdict
import pandas as pd
from context_manager import ContextManager

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

def evaluate_sample(sample, manager, modes, top_k, token_budget, use_reranker, top_k_ratio):
    conversation = sample.get("conversation", [])
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

    results = {}
    available_chunks = len(manager.session.rag.chunks)
    ratio_k = max(1, int(round(available_chunks * top_k_ratio))) if available_chunks else top_k
    effective_top_k = min(top_k, ratio_k) if available_chunks else top_k
    for mode in modes:
        rerank = use_reranker and mode.endswith("_rerank")
        base_mode = mode.replace("_rerank", "")
        pkg = manager.build_prompt_context(
            query=query,
            token_budget=token_budget,
            mode=base_mode,
            top_k=effective_top_k,
            use_reranker=rerank,
        )
        results[mode] = {
            "context_tokens": pkg.get("context_tokens", 0),
        }

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/querycontext_dataset.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=600)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k-ratio", type=float, default=0.6)
    parser.add_argument("--reranker-weights", default="train/model.pth")
    parser.add_argument("--modes", nargs="+", default=["full_context", "rag", "dynamic_context", "rag_rerank", "dynamic_context_rerank"])
    parser.add_argument("--output", default="results/token_reduction.csv")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        dataset = json.load(f)

    manager = ContextManager(reranker_weights=args.reranker_weights)
    totals = defaultdict(lambda: {"tokens": 0.0, "count": 0})
    baseline_tokens = []

    for sample in dataset[: args.limit]:
        result = evaluate_sample(
            sample,
            manager,
            args.modes,
            args.top_k,
            args.token_budget,
            use_reranker=True,
            top_k_ratio=args.top_k_ratio,
        )
        if result is None:
            continue

        full_tokens = result.get("full_context", {}).get("context_tokens", 0)
        if full_tokens > 0:
            baseline_tokens.append(full_tokens)

        for mode, metrics in result.items():
            totals[mode]["tokens"] += metrics.get("context_tokens", 0)
            totals[mode]["count"] += 1

    rows = []
    baseline_avg = sum(baseline_tokens) / len(baseline_tokens) if baseline_tokens else 0.0

    for mode in args.modes:
        count = totals[mode]["count"]
        if count == 0:
            continue
        avg_tokens = totals[mode]["tokens"] / count
        reduction_pct = 0.0
        if baseline_avg > 0:
            reduction_pct = max(0.0, (1.0 - (avg_tokens / baseline_avg)) * 100.0)

        rows.append(
            {
                "mode": mode,
                "avg_context_tokens": avg_tokens,
                "reduction_pct_vs_full": reduction_pct,
                "samples": count,
            }
        )

    df = pd.DataFrame(rows).sort_values("mode").reset_index(drop=True)
    print("\nToken Reduction Results")
    print(df.to_string(index=False))

    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved results to: {output_path}")

if __name__ == "__main__":
    main()