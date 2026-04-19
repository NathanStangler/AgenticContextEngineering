import torch
import math
from rag import RAG
from model.cross_transformer import MultiChunkTransformer
import random
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MultiChunkTransformer().to(device)
model.load_state_dict(torch.load("train/model.pth", map_location=device))
model.eval()

wildchat_ds = [
    {
        "id": "conv_1",
        "conversation": [
            {"role": "user",      "content": "How do I reverse a string in Python?"},
            {"role": "assistant", "content": "You can reverse a string in Python using slicing: my_str[::-1]. This returns a new reversed string without modifying the original."},
            {"role": "user",      "content": "What about reversing a list?"},
            {"role": "assistant", "content": "To reverse a list, use my_list.reverse() to modify in place, or list(reversed(my_list)) to get a new reversed list. You can also use slicing: my_list[::-1]."},
            {"role": "user",      "content": "How do I sort a list in descending order?"},
        ],
    },
    {
        "id": "conv_2",
        "conversation": [
            {"role": "user",      "content": "What is the difference between supervised and unsupervised learning?"},
            {"role": "assistant", "content": "Supervised learning trains on labeled data — you provide inputs and correct outputs, and the model learns the mapping. Examples: classification, regression."},
            {"role": "assistant", "content": "Unsupervised learning finds hidden patterns in unlabeled data. The model has no ground-truth labels. Examples: clustering, dimensionality reduction, autoencoders."},
            {"role": "user",      "content": "Can you give an example of a supervised learning algorithm?"},
            {"role": "assistant", "content": "A common supervised learning algorithm is logistic regression, used for binary classification. Random forests and SVMs are also popular supervised methods."},
            {"role": "user",      "content": "What is reinforcement learning?"},
        ],
    },
    {
        "id": "conv_3",
        "conversation": [
            {"role": "user",      "content": "What causes climate change?"},
            {"role": "assistant", "content": "Climate change is primarily driven by the increase of greenhouse gases like CO2 and methane in the atmosphere, largely from burning fossil fuels, deforestation, and industrial activity."},
            {"role": "user",      "content": "What are the effects of climate change?"},
            {"role": "assistant", "content": "Effects include rising sea levels, more frequent extreme weather events, shifting ecosystems, loss of biodiversity, and disruptions to agriculture and water supply."},
            {"role": "user",      "content": "How can individuals reduce their carbon footprint?"},
            {"role": "assistant", "content": "Individuals can reduce carbon footprint by using public transport, eating less meat, reducing energy consumption at home, flying less, and buying locally produced goods."},
            {"role": "user",      "content": "What are renewable energy sources?"},
        ],
    },
    {
        "id": "conv_4",
        "conversation": [
            {"role": "user",      "content": "How does HTTP work?"},
            {"role": "assistant", "content": "HTTP (HyperText Transfer Protocol) is a request-response protocol. A client sends a request (GET, POST, etc.) to a server, which processes it and returns a response with a status code and body."},
            {"role": "user",      "content": "What is the difference between GET and POST?"},
            {"role": "assistant", "content": "GET requests retrieve data and append parameters in the URL. POST requests send data in the request body, suitable for submitting forms or uploading data. GET is idempotent; POST is not."},
            {"role": "user",      "content": "What does a 404 error mean?"},
        ],
    },
    {
        "id": "conv_5",
        "conversation": [
            {"role": "user",      "content": "What is a transformer model in NLP?"},
            {"role": "assistant", "content": "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequences. It was introduced in 'Attention Is All You Need' (2017) and is the basis of models like BERT and GPT."},
            {"role": "user",      "content": "What is self-attention?"},
            {"role": "assistant", "content": "Self-attention allows each token in a sequence to attend to all other tokens. It computes query, key, and value vectors and uses dot-product attention to weight how much each token influences the others."},
            {"role": "user",      "content": "How is BERT different from GPT?"},
        ],
    },
]

def build_corpus(samples):
    corpus = []

    for s in samples:
        for t in s["conversation"]:
            corpus.append(t["content"])

    return list(set(corpus)) 


def build_test_cases(samples):
    test_cases = []

    for s in samples:
        turns = s["conversation"]

        if turns[-1]["role"] != "user":
            continue

        query = turns[-1]["content"]

        relevant = {
            t["content"]
            for t in turns
            if t["role"] == "assistant"
        }

        test_cases.append({
            "id": s["id"],
            "query": query,
            "relevant": relevant
        })

    return test_cases


def dcg(rels, k):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(preds, relevant, k=5):
    rels = [1 if p in relevant else 0 for p in preds]
    ideal = sorted(rels, reverse=True)

    dcg_val = dcg(rels, k)
    idcg_val = dcg(ideal, k)

    return dcg_val / idcg_val if idcg_val > 0 else 0


def recall(preds, relevant, k=5):
    if not relevant:
        return 0.0
    return sum(p in relevant for p in preds[:k]) / len(relevant)

def rerank(model, query, chunks):
    with torch.no_grad():
        scores = model([query], [chunks])[0]
        idx = torch.argsort(scores, descending=True).tolist()
    return [chunks[i] for i in idx]


def run_eval(k=5):
    corpus = build_corpus(wildchat_ds)
    test_cases = build_test_cases(wildchat_ds)

    ndcg_base, ndcg_rerank = [], []
    rec_base, rec_rerank = [], []

    print("Running corrected RAG evaluation...\n")

    for tc in test_cases:
        query = tc["query"]
        relevant = tc["relevant"]

        rag = RAG()

        for c in random.sample(corpus, min(200, len(corpus))):
            rag.add_context(c)

        retrieved = rag.retrieve_top_k(query, k=10)
        chunks = [x["text"] for x in retrieved]

        if not chunks:
            continue

        reranked = rerank(model, query, chunks)


        ndcg_base.append(ndcg(chunks, relevant, k))
        ndcg_rerank.append(ndcg(reranked, relevant, k))

        rec_base.append(recall(chunks, relevant, k))
        rec_rerank.append(recall(reranked, relevant, k))

        print(f"[{tc['id']}] {query}")
        print(f"  NDCG  base={ndcg_base[-1]:.4f}  rerank={ndcg_rerank[-1]:.4f}")
        print(f"  Recall base={rec_base[-1]:.4f}  rerank={rec_rerank[-1]:.4f}\n")

    print("\n===== FINAL =====")
    print(f"NDCG@{k}")
    print(f"  Base:   {sum(ndcg_base)/len(ndcg_base):.4f}")
    print(f"  Rerank: {sum(ndcg_rerank)/len(ndcg_rerank):.4f}")

    print(f"\nRecall@{k}")
    print(f"  Base:   {sum(rec_base)/len(rec_base):.4f}")
    print(f"  Rerank: {sum(rec_rerank)/len(rec_rerank):.4f}")


if __name__ == "__main__":
    run_eval(k=5)