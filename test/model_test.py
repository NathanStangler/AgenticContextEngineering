import torch
import math
from torch.utils.data import DataLoader, random_split

from model.cross_transformer import MultiChunkTransformer
from data.processed.rag_test_dataset import ContextEngDataset
from data.processed.data_utils import collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MultiChunkTransformer().to(device)
model.load_state_dict(torch.load("train/model.pth", map_location=device))
model.eval()

dataset = ContextEngDataset("data/processed/querycontext_dataset.json")

test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size

_, test_ds = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn
)


def dcg_at_k(relevances, k):
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        rel = relevances[i]
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(model, loader, device, k=5):
    model.eval()
    total_ndcg = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            scores_batch = model(batch["conversation"], batch["chunks"])
            labels_batch = batch["labels"]

            for i in range(len(scores_batch)):
                scores_i = scores_batch[i].detach()

                labels_i = torch.tensor(
                    labels_batch[i],
                    device="cpu"
                )

                # 🔥 FIX: force same length
                L = min(len(scores_i), len(labels_i))

                scores_i = scores_i[:L]
                labels_i = labels_i[:L]

                sorted_idx = torch.argsort(scores_i, descending=True)

                ranked_rels = labels_i[sorted_idx].tolist()
                ideal_rels = sorted(labels_i.tolist(), reverse=True)

                dcg = dcg_at_k(ranked_rels, k)
                idcg = dcg_at_k(ideal_rels, k)

                ndcg = dcg / idcg if idcg > 0 else 0.0

                total_ndcg += ndcg
                total += 1

    return total_ndcg / max(total, 1)


def recall_at_k(model, loader, device, k=5, threshold=0.01):
    model.eval()
    total_recall = 0.0
    num_queries = 0

    with torch.no_grad():
        for batch in loader:
            scores_batch = model(batch["conversation"], batch["chunks"])
            labels_batch = batch["labels"]

            for i in range(len(scores_batch)):
                scores_i = scores_batch[i].detach()

                labels_i = torch.tensor(labels_batch[i], device="cpu")

                L = min(len(scores_i), len(labels_i))
                scores_i = scores_i[:L]
                labels_i = labels_i[:L]

                topk_idx = torch.argsort(scores_i, descending=True)[:k]

                relevant = labels_i > threshold
                num_relevant = relevant.sum().item()

                if num_relevant == 0:
                    continue

                hits = relevant[topk_idx].sum().item()

                total_recall += hits / num_relevant
                num_queries += 1

    return total_recall / max(num_queries, 1)


def get_ranked_chunks(model, q_batch, chunks_batch, device):
    model.eval()

    with torch.no_grad():
        scores_batch = model(q_batch, chunks_batch)

        results = []

        for i in range(len(scores_batch)):
            scores_i = scores_batch[i]

            sorted_idx = torch.argsort(scores_i, descending=True).cpu().tolist()

            valid_len = len(chunks_batch[i])
            sorted_idx = [j for j in sorted_idx if j < valid_len]

            ranked_chunks = [chunks_batch[i][j] for j in sorted_idx]
            ranked_scores = scores_i[sorted_idx].tolist()

            results.append({
                "ranked_chunks": ranked_chunks,
                "scores": ranked_scores
            })

    return results


def run_test():
    print("Running evaluation...\n")

    ndcg = ndcg_at_k(model, test_loader, device)
    recall = recall_at_k(model, test_loader, device)

    print("\nModel Testing Results")
    print(f"NDCG@K:  {ndcg:.4f}")
    print(f"Recall@K:{recall:.4f}\n")

    sample_batch = next(iter(test_loader))

    ranked = get_ranked_chunks(
        model,
        sample_batch["conversation"],
        sample_batch["chunks"],
        device
    )

    for i in range(min(2, len(ranked))):
        print(f"\nQUERY {i+1}: {sample_batch['conversation'][i]}\n")

        for j, (chunk, score) in enumerate(
            zip(ranked[i]["ranked_chunks"], ranked[i]["scores"])
        ):
            print(f"Rank {j+1} | Score: {score:.4f}")
            print(chunk, "\n")

    return {
        "NDCG@K": ndcg,
        "Recall@K": recall
    }


if __name__ == "__main__":
    results = run_test()
    print(results)