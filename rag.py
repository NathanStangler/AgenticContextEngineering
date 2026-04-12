import faiss
from sentence_transformers import SentenceTransformer

class RAG:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.index = faiss.IndexFlatIP(self.model.get_embedding_dimension())

    def encode(self, chunks):
        vectors = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        # faiss.normalize_L2(vectors)
        return vectors

    def add_context(self, context):
        chunks, vectors = self.chunk_context(context)

        if not chunks:
            return

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def retrieve_top_k(self, query, k=5):
        if not self.chunks:
            return []

        k = max(1, min(k, len(self.chunks)))
        query_vector = self.encode([query])
        scores, indices = self.index.search(query_vector, k)
        scores, indices = scores[0], indices[0]

        results = []
        for score, index in zip(scores, indices):
            if index < 0 or index >= len(self.chunks):
                continue
            results.append({"text": self.chunks[index], "score": float(score)})
        return results

    def chunk_context(self, context):
        chunk_tokens = self.model.tokenizer.model_max_length
        overlap_tokens = int(0.2 * chunk_tokens)

        context = context.strip()
        if not context:
            return [], None

        token_ids = self.model.tokenizer.encode(context, add_special_tokens=False)
        chunks = []
        for i in range(0, len(token_ids), chunk_tokens - overlap_tokens):
            ids = token_ids[i : i + chunk_tokens]
            if not ids:
                continue
            text = self.model.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            if len(text.split()) < 3:
                continue
            if text:
                chunks.append(text)

        if not chunks:
            return [], None
        return chunks, self.encode(chunks)
