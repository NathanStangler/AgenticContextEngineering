import torch
import math
import random
from rag import RAG
from model.cross_transformer import MultiChunkTransformer
from summarization import Summarization
from transformers import AutoTokenizer
from token_usage import count_tokens



device = "cuda" if torch.cuda.is_available() else "cpu"

model = MultiChunkTransformer().to(device)
model.load_state_dict(torch.load("train/model.pth", map_location=device))
model.eval()

summarizer = Summarization()

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")


wildchat_ds = [
    {
        "id": "conv_1",
        "conversation": [
            {"role": "user", "content": "How do I reverse a string in Python?"},
            {"role": "assistant", "content": "You can reverse a string in Python using slicing: my_str[::-1]. This returns a new reversed string without modifying the original."},
            {"role": "user", "content": "What about reversing a list?"},
            {"role": "assistant", "content": "To reverse a list, use my_list.reverse() to modify in place, or list(reversed(my_list)) to get a new reversed list. You can also use slicing: my_list[::-1]."},
            {"role": "user", "content": "How do I sort a list in descending order?"},
        ],
    },
    {
        "id": "conv_2",
        "conversation": [
            {"role": "user", "content": "What is the difference between supervised and unsupervised learning?"},
            {"role": "assistant", "content": "Supervised learning trains on labeled data..."},
            {"role": "assistant", "content": "Unsupervised learning finds hidden patterns..."},
            {"role": "user", "content": "Can you give an example of a supervised learning algorithm?"},
            {"role": "assistant", "content": "A common supervised learning algorithm is logistic regression..."},
            {"role": "user", "content": "What is reinforcement learning?"},
        ],
    },
]

def build_corpus(samples):
    return list({
        t["content"]
        for s in samples
        for t in s["conversation"]
    })


def build_test_cases(samples):
    test_cases = []

    for s in samples:
        turns = s["conversation"]

        if turns[-1]["role"] != "user":
            continue

        test_cases.append({
            "id": s["id"],
            "query": turns[-1]["content"],
            "relevant": {
                t["content"] for t in turns if t["role"] == "assistant"
            }
        })

    return test_cases


# Metrics

def dcg(rels, k):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(preds, relevant, k=5):
    rels = [1 if p in relevant else 0 for p in preds]
    ideal = sorted(rels, reverse=True)
    return dcg(rels, k) / dcg(ideal, k) if dcg(ideal, k) > 0 else 0


def recall(preds, relevant, k=5):
    if not relevant:
        return 0.0
    return sum(p in relevant for p in preds[:k]) / len(relevant)


def answer_hit(answer, relevant):
    answer = answer.lower()
    return any(r.lower() in answer for r in relevant)




def rerank(model, query, chunks):
    with torch.no_grad():
        scores = model([query], [chunks])[0]
        idx = torch.argsort(scores, descending=True).tolist()
    return [chunks[i] for i in idx]


def rag_pipeline(query, corpus, k=5):
    rag = RAG()

    for c in corpus:
        rag.add_context(c)

    retrieved = rag.retrieve_top_k(query, k=10)
    chunks = [x["text"] for x in retrieved]

    if not chunks:
        return None

    reranked = rerank(model, query, chunks)
    top_chunks = reranked[:k]

    # Query-aware summarization (IMPORTANT)
    combined = f"Question: {query}\n\nContext:\n" + "\n".join(top_chunks)
    answer = summarizer.generate(combined, max_new_tokens=128, min_new_tokens=30)

    return {
        "retrieved": chunks,
        "reranked": reranked,
        "top_chunks": top_chunks,
        "answer": answer
    }


#. Eval Test
def run_eval(k=5):
    corpus = build_corpus(wildchat_ds)
    test_cases = build_test_cases(wildchat_ds)

    ndcg_base, ndcg_rerank = [], []
    rec_base, rec_rerank = [], []
    answer_scores = []

    total_input_tokens = []
    total_output_tokens = []

    print("Running FULL RAG pipeline...\n")

    for tc in test_cases:
        query = tc["query"]
        relevant = tc["relevant"]

        sampled_corpus = random.sample(corpus, min(200, len(corpus)))

        result = rag_pipeline(query, sampled_corpus, k)

        if result is None:
            continue

        chunks = result["retrieved"]
        reranked = result["reranked"]
        top_chunks = result["top_chunks"]
        answer = result["answer"]

        # Retrieval metrics
        ndcg_base.append(ndcg(chunks, relevant, k))
        ndcg_rerank.append(ndcg(reranked, relevant, k))

        rec_base.append(recall(chunks, relevant, k))
        rec_rerank.append(recall(reranked, relevant, k))

        # Answer quality
        hit = answer_hit(answer, relevant)
        answer_scores.append(hit)

        # Token tracking
        input_tokens = sum(count_tokens(c) for c in top_chunks)
        output_tokens = count_tokens(answer)

        total_input_tokens.append(input_tokens)
        total_output_tokens.append(output_tokens)

        print(f"[{tc['id']}] {query}")
        print(f"Answer: {answer}")
        print(f"Hit: {hit}")
        print(f"NDCG base={ndcg_base[-1]:.3f} rerank={ndcg_rerank[-1]:.3f}")
        print(f"Recall base={rec_base[-1]:.3f} rerank={rec_rerank[-1]:.3f}")
        print(f"Tokens: in={input_tokens} out={output_tokens}\n")

    print("\n===== FINAL =====")

    print(f"NDCG@{k}")
    print(f"  Base:   {sum(ndcg_base)/len(ndcg_base):.4f}")
    print(f"  Rerank: {sum(ndcg_rerank)/len(ndcg_rerank):.4f}")

    print(f"\nRecall@{k}")
    print(f"  Base:   {sum(rec_base)/len(rec_base):.4f}")
    print(f"  Rerank: {sum(rec_rerank)/len(rec_rerank):.4f}")

    print(f"\nAnswer Hit Rate: {sum(answer_scores)/len(answer_scores):.4f}")

    avg_in = sum(total_input_tokens) / len(total_input_tokens)
    avg_out = sum(total_output_tokens) / len(total_output_tokens)

    print(f"\nAvg Input Tokens: {avg_in:.2f}")
    print(f"Avg Output Tokens: {avg_out:.2f}")
    print(f"Compression Ratio: {avg_out / max(avg_in,1):.3f}")



if __name__ == "__main__":
    run_eval(k=5)