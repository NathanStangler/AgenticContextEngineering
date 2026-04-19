from transformers import AutoTokenizer

def count_tokens(text, model_name="google-t5/t5-base"):
    if not text:
        return 0

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))


# import torch
# import torch.nn as nn
# from torch import Generator
# from torch.utils.data import DataLoader, Subset
# from torch.nn.utils import clip_grad_norm_
# from collections import defaultdict
# import random
# import torch.nn.functional as F
# from data.processed.rag_test_dataset import ContextEngDataset
# from model.cross_transformer import MultiChunkTransformer

# def collate_fn(batch): # Change from using one item from batch to feeding in the batch itself.
#     return {
#         "query": [item["query"] for item in batch],
#         "chunks": [item["chunks"] for item in batch],
#         "labels": torch.stack([item["labels"] for item in batch]), # Labels are mimicing the chunk importance scores
#     }


# def listwise_ranking_func(chunk_scores, labels, temp):

#     labels = F.softmax(labels, dim=-1)
#     chunk_scores = (chunk_scores - chunk_scores.mean(dim=-1, keepdim=True))
#     dist_labels = F.softmax(labels / temp, dim=-1)
#     dist_chunk_scores = F.log_softmax(chunk_scores / temp, dim=-1)
    
#     kl_loss = F.kl_div(dist_chunk_scores, dist_labels, reduction='batchmean')
#     return kl_loss

# def build_query_split(dataset, train_ratio=0.8, seed=42): # Split by query for training and testing to get different queries in each.
#     query_to_indices = defaultdict(list)

#     for i in range(len(dataset)):
#         q = dataset[i]["query"]
#         query_to_indices[q].append(i)

#     all_queries = list(query_to_indices.keys())
#     random.seed(seed)
#     random.shuffle(all_queries)

#     split_idx = int(train_ratio * len(all_queries))

#     train_queries = set(all_queries[:split_idx])
#     val_queries = set(all_queries[split_idx:])

#     train_indices = []
#     val_indices = []

#     for q in train_queries:
#         train_indices.extend(query_to_indices[q])

#     for q in val_queries:
#         val_indices.extend(query_to_indices[q])

#     return train_indices, val_indices

# dataset = ContextEngDataset("data/processed/querycontext_dataset.json")
# train_split_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_split_size

# train_idx, val_idx = build_query_split(dataset=dataset)
# train_ds, val_ds = Subset(dataset=dataset, indices=train_idx), Subset(dataset=dataset, indices=val_idx)

# seed_gen = Generator()
# seed_gen.manual_seed(42)

# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn, generator=seed_gen)
# val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, generator=seed_gen)

# attention_model = MultiChunkTransformer()
# optimizer = torch.optim.AdamW(attention_model.parameters(), lr=1e-4,  weight_decay=0.01)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# num_epochs = 10
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

# def train_model(model, train_loader, optimizer, device, epochs, margin=1.0):
#     model.to(device)

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         temp = max(0.5, 2.0 * (0.95 ** epoch)) # Tune temperature for better training score decrease.

#         for batch in train_loader:
#             q_batch = batch["query"]
#             chunks_batch = batch["chunks"]
#             labels_batch = batch["labels"].to(device)

#             optimizer.zero_grad()
#             batch_loss = 0.0

#             score_list = [model([q_batch[i]], [chunks_batch[i]])[0] for i in range(len(q_batch))]
#   # Based on the number of chunks

#             scores = torch.stack(score_list, dim=0)
#             labels = labels_batch
#             loss = listwise_ranking_func(scores, labels, temp)


#             batch_loss = loss

#             batch_loss.backward()
#             clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()

#             total_loss += batch_loss.item()

#         scheduler.step()
#         val_loss = evaluate_model(model, val_loader, device, temp=1.0)

#         print(
#             f"Epoch {epoch+1}/{epochs} | "
#             f"Train Loss: {total_loss / len(train_loader):.6f} | "
#             f"Val Loss: {val_loss:.6f}"
#         )


# # Evaluation
# def evaluate_model(model, val_loader, device, temp):
#     model.to(device)
#     model.eval()

#     total_loss = 0.0

#     with torch.no_grad():
#         for batch in val_loader:
#             q_batch = batch["query"]
#             chunks_batch = batch["chunks"]
#             labels_batch = batch["labels"].to(device)

#             batch_loss = 0.0
#             score_list = [model([q_batch[i]], [chunks_batch[i]])[0] for i in range(len(q_batch))]
#              # Based on the number of chunks

#             scores = torch.stack(score_list, dim=0)
#             labels = labels_batch
#             loss = listwise_ranking_func(scores, labels, temp)

#             batch_loss = loss

#             total_loss += batch_loss.item()

#     return total_loss / len(val_loader)

# def check_data_leakage(train_ds, val_ds): # Check for data leakage when splitting and shuffling.
#     train_set = set()
#     val_set = set()

#     for i in range(len(train_ds)):
#         train_set.add(str(train_ds[i]["query"]))

#     for i in range(len(val_ds)):
#         val_set.add(str(val_ds[i]["query"]))

#     overlap = train_set & val_set
#     print("Query overlap:", len(overlap))
#     return overlap



# data_overlap = check_data_leakage(train_ds, val_ds)
# train_model(model=attention_model, train_loader=train_loader, optimizer=optimizer, device=device, epochs=num_epochs)
# torch.save(attention_model.state_dict(), "train/model.pth")
# print("Val Loss: ", evaluate_model(model=attention_model, val_loader=val_loader, device=device))