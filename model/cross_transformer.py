import runtime_settings
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
from torch.nn.utils.rnn import pad_sequence


class MultiChunkTransformer(nn.Module):
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", device='cpu'):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.encoder = AutoModel.from_pretrained(model) # Encode top_k chunk from RAG output.
        for name, param in self.encoder.named_parameters():
            if "encoder.layer.5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        hidden_layer_dim = self.encoder.config.hidden_size

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_layer_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )

        self.query_gate = nn.Linear(hidden_layer_dim, hidden_layer_dim)

        self.scoring_nn = nn.Sequential(
            nn.LayerNorm(hidden_layer_dim),
            nn.Linear(hidden_layer_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
        
        self.to(device)

    def encode(self, txt):
        if isinstance(txt, str):
            txt = [txt]
        ids = [
            self.tokenizer.encode(str(t), truncation=True, max_length=256)
            for t in txt
        ]
        max_len = max(len(e) for e in ids)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.tensor(
            [e + [pad_id] * (max_len - len(e)) for e in ids]
        ).to(self.device)
        attention_mask = torch.tensor(
            [[1] * len(e) + [0] * (max_len - len(e)) for e in ids]
        ).to(self.device)
        encode_output_pairs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb_txt = encode_output_pairs.last_hidden_state[:, 0, :]
        return cls_emb_txt

    def conversation_to_text(self, conv): # Convert our conversation to text for transformer to 
        if isinstance(conv, str):
            return conv

        if not isinstance(conv, list):
            return str(conv)

        if not conv:
            return ""

        recent_turns = conv[-4:]

        if all(isinstance(turn, dict) for turn in recent_turns):
            return " ".join(
                turn.get("content", "")
                for turn in recent_turns
                if turn.get("content")
            )

        return " ".join(str(turn) for turn in recent_turns if str(turn))

    
    def forward(self, conversation, chunks): # List of conversations, chunks


        B = len(conversation)
        convo_texts = [
            self.conversation_to_text(conv) for conv in conversation
        ]

        C = self.encode(convo_texts).unsqueeze(1)

        flat_chunks = [c for group in chunks for c in group]

        H_1d = self.encode(flat_chunks)

        lengths = [len(c) for c in chunks]
        H = torch.split(H_1d, lengths)

        H = pad_sequence(H, batch_first=True) 

        attn_out, _ = self.cross_attention(H, C, C) 
        H = H + attn_out


        gate_input = H + C.expand(-1, H.shape[1], -1)
        gate = torch.sigmoid(self.query_gate(gate_input))

        H = H * gate

        return self.scoring_nn(H).squeeze(-1)

# Need to do training and inference


