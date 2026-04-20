import json 
import torch 
from torch.utils.data import Dataset

class ContextEngDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        conversation = item["conversation"] # Agent sees

        # retrieval candidates
        candidates = item["retrieval_candidates"]

        texts = [
            c["text"] if isinstance(c, dict) else c
            for c in candidates
        ]

        importance = torch.tensor(
            [c.get("importance", 0.0) for c in candidates],
            dtype=torch.float32
        )

        selected_context = torch.tensor(
            item.get("selected_context", []),
            dtype=torch.long
        )

        token_budget = torch.tensor(
            item.get("token_budget", 0),
            dtype=torch.long
        )

        return {
            "conversation": conversation,
            "chunks": texts,
            "importance": importance,
            "selected_context": selected_context,
            "token_budget": token_budget
        }