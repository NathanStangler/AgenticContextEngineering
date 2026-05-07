from pathlib import Path
import torch
from model.cross_transformer import MultiChunkTransformer

class LearnedChunkReranker:
    def __init__(self, weights_path="train/model.pth", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.weights_path = Path(weights_path) if weights_path else None
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.available = False

        try:
            self.model = MultiChunkTransformer(model=model_name, device=self.device)
            if self.weights_path and self.weights_path.exists():
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.available = True
        except Exception:
            self.model = None
            self.available = False

    def score(self, query, chunks):
        if not chunks:
            return []

        if not self.available or self.model is None:
            return [0.0 for _ in chunks]

        with torch.no_grad():
            scores = self.model([query], [list(chunks)])[0]
        return scores.detach().cpu().tolist()

    def rerank(self, query, chunks):
        if len(chunks) <= 1:
            return list(chunks)

        scores = self.score(query, chunks)
        order = sorted(range(len(chunks)), key=lambda idx: scores[idx], reverse=True)
        return [chunks[idx] for idx in order]

    def rerank_with_scores(self, query, chunks):
        scores = self.score(query, chunks)
        order = sorted(range(len(chunks)), key=lambda idx: scores[idx], reverse=True)
        return [(chunks[idx], float(scores[idx])) for idx in order]