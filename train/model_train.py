import torch
import torch.nn as nn
from torch import Generator
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
import random
import torch.nn.functional as F
from data.processed.data_utils import collate_fn, pad_to_model_len, flatten_batch
from tqdm import tqdm
from data.processed.rag_test_dataset import ContextEngDataset
from model.cross_transformer import MultiChunkTransformer

 # Comparing lists (scores, labels) to rank - do random for efficency instead of looping through both data
def listwise_loss(scores, labels, lengths):
    temp = 1.0
    loss = 0

    for b in range(len(scores)):
        k = lengths[b]
        s = scores[b, :k]
        l = labels[b, :k]

        l = (l - l.mean()) / (l.std() + 1e-8)

        probs = torch.softmax(s / temp, dim=0)
        target = torch.softmax(l / temp, dim=0)

        loss += F.kl_div(probs.log(), target, reduction="batchmean")

    return loss / len(scores)

def build_conversation_split(dataset, train_ratio=0.8, seed=42): # Split by conv for training and testing to get different conv in each.
    conversation_to_indices = defaultdict(list)

    for i in range(len(dataset)):
        user_turns = tuple(
            turn["content"]
            for turn in dataset[i]["conversation"]
            if turn["role"] == "user" and turn["content"]
        )
        if user_turns:
            conversation_to_indices[user_turns].append(i)

    all_conversations = list(conversation_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(all_conversations)

    split_idx = int(train_ratio * len(all_conversations))

    train_conversations = set(all_conversations[:split_idx])
    val_conversations = set(all_conversations[split_idx:])

    train_indices = []
    val_indices = []

    for convo in train_conversations:
        train_indices.extend(conversation_to_indices[convo])

    for convo in val_conversations:
        val_indices.extend(conversation_to_indices[convo])

    return train_indices, val_indices

dataset = ContextEngDataset("data/processed/querycontext_dataset.json")
train_split_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_split_size

train_idx, val_idx = build_conversation_split(dataset=dataset)
train_ds, val_ds = Subset(dataset=dataset, indices=train_idx), Subset(dataset=dataset, indices=val_idx)

seed_gen = Generator()
seed_gen.manual_seed(42)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn, generator=seed_gen)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, generator=seed_gen)

attention_model = MultiChunkTransformer()
optimizer = torch.optim.AdamW(attention_model.parameters(), lr=1e-4, weight_decay=0.001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 10
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.5)

def train_model(model, train_loader, optimizer, device, epochs, margin=1.0):
    model.to(device)

    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0

        for batch in train_loader:
            conv_batch = batch["conversation"]
            chunks_batch = batch["chunks"]
            labels_batch = batch["labels"]


            optimizer.zero_grad()

            lengths = [len(c) for c in chunks_batch]
            scores = model(conv_batch, chunks_batch)
            scores_pad, label_pad = pad_to_model_len(scores, labels_batch, device)
            loss =  listwise_loss(scores_pad, label_pad, lengths=lengths)

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, device)
        scheduler.step(val_loss) # Want to monitor validation loss and scheduler adjusts accordingly.

        train_loss = total_loss / len(train_loader)
        epoch_bar.set_postfix(train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}")
        tqdm.write(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")


# Evaluation
def evaluate_model(model, val_loader, device):
    model.to(device)
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            conv_batch = batch["conversation"]
            chunks_batch = batch["chunks"]
            labels_batch = batch["labels"]

            lengths = [len(c) for c in chunks_batch]
            scores = model(conv_batch, chunks_batch)
            scores_pad, label_pad = pad_to_model_len(scores, labels_batch, device)
            loss =  listwise_loss(scores_pad, label_pad, lengths=lengths)

            total_loss += loss.item()

    return total_loss / len(val_loader)

def check_data_leakage(train_ds, val_ds): # Check for data leakage when splitting and shuffling.
    train_set = set()
    val_set = set()

    for i in range(len(train_ds)):
        train_set.add(str(train_ds[i]["conversation"]))

    for i in range(len(val_ds)):
        val_set.add(str(val_ds[i]["conversation"]))

    overlap = train_set & val_set
    print("Agent Conversation overlap:", len(overlap))
    return overlap



data_overlap = check_data_leakage(train_ds, val_ds)
train_model(model=attention_model, train_loader=train_loader, optimizer=optimizer, device=device, epochs=num_epochs)
torch.save(attention_model.state_dict(), "train/model.pth")
print("Val Loss: ", evaluate_model(model=attention_model, val_loader=val_loader, device=device))
