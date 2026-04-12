# Notes:
# Multi-Chunk Cross-Attention Transformer for Chunk Importance Scoring
# Approach:
# Encode the Top_k chunks from RAG output -> End with (query, chunk1), ... (query, chunkn)
# Stack/Concatenate those q,c pairs and feed into a cross attention transformer
# - Self-attend to the query for each chunk in chunks
# Have a Attention mask so that the model learns what it should be attending to 
# Gate (Layer) for controlling how important each chunk is with respect to the query (query-conditional)
# Scoring Logic - Use Normal Feed-forward


import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
class MultiChunkTransformer(nn.Module):
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", device='cpu'):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.encoder = AutoModel.from_pretrained(model) # Encode top_k chunk from RAG output.

        hidden_layer_dim = self.encoder.config.hidden_size

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_layer_dim,
            num_heads=8,
            batch_first=True
        )

        self.query_gate = nn.Linear(hidden_layer_dim, hidden_layer_dim)

        self.scoring_nn = nn.Sequential(
            nn.Linear(hidden_layer_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.to(device)

    def encode_query(self, txt):
        inputs = self.tokenizer(
        txt,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding=True
        ).to(self.device)
        encode_output_pairs = self.encoder(**inputs)
        cls_emb_txt = encode_output_pairs.last_hidden_state[:, 0, :] # Get CLS (Embedding of start of input chunk sequence)
        return cls_emb_txt

    def encode_rag_outputs(self, rag_chunk):
        inputs = self.tokenizer(
            rag_chunk,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)
        encode_output_pairs = self.encoder(**inputs)
        return encode_output_pairs.last_hidden_state[:, 0, :] # Get CLS (Embedding of start of input chunk sequence)
    
    def forward(self, query, chunks):
        Q = self.encode_query(query)
        H = self.encode_rag_outputs(chunks)
        H = H.unsqueeze(0)
        
        Q_attention = Q.unsqueeze(1)
        H = torch.cat([Q_attention, H], dim=1)
        H, _ = self.cross_attention(H, H, H) # Don't need attention weights here, just the outputs (Do Q, K, V relative to RAG chunks)

        H = H[:, 1:, :] # Want query out of the chunks

        gate = torch.sigmoid(self.query_gate(Q))
        H = H * gate.unsqueeze(1) 

        chunk_scores = self.scoring_nn(H).squeeze(-1) # (1 batch size and K RAG outputs)
        importance_scores = torch.softmax(chunk_scores, dim=1)
        return importance_scores

# Need to do training and inference




