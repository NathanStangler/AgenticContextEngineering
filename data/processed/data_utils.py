import torch

def collate_fn(batch): # Change from using one item from batch to feeding in the batch itself.
    return {
        "conversation": [item["conversation"] for item in batch],
        "chunks": [item["chunks"] for item in batch],
        "labels": [item["importance"] for item in batch], # Labels are mimicing the chunk importance scores
        "token_budget": torch.stack([item["token_budget"] for item in batch])
    }


def flatten_batch(q_batch, chunks_batch): # Batch gets processed once instead of each time to improve epoch times.
    queries = []
    chunks = []
    query_idx = []

    for i in range(len(q_batch)):
        q = q_batch[i]
        chunk_list = chunks_batch[i]

        for c in chunk_list:
            queries.append(q)
            chunks.append(c)
            query_idx.append(i)

    return queries, chunks, query_idx

def pad_to_model_len(scores, labels, device):
    B, K_max = scores.shape
    padded_labels = []
    for l_list in labels:
        l = torch.as_tensor(l_list, dtype=torch.float, device=device)
        pad_size = K_max - l.shape[0]
        if pad_size > 0:
            l = torch.cat([l, torch.zeros(pad_size, device=device)], dim=0)
        padded_labels.append(l)
    return scores, torch.stack(padded_labels)
