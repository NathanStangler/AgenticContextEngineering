import faiss
from sentence_transformers import SentenceTransformer


class RAG:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", local_files_only=True):
        self.model = SentenceTransformer(
            embedding_model,
            local_files_only=local_files_only, # Load Model Locally.
        )
        self.model.max_seq_length = 200
        self.chunks = []
        self.index = faiss.IndexFlatIP(self.model.get_embedding_dimension())

    def encode(self, chunks):
        vectors = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        # faiss.normalize_L2(vectors)
        return vectors

    def reset(self):
        self.chunks = []
        self.index = faiss.IndexFlatIP(self.model.get_embedding_dimension())

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

    def chunk_context(self, context, max_tokens=200, overlap_percentage=0.2):
        chunk_tokenizer = self.model.tokenizer
        overlap_tokens = int(max_tokens * overlap_percentage)

        context = context.strip()
        if not context:
            return [], None

        def recursive_splitting(txt):
            token_length = len(chunk_tokenizer.encode(txt, add_special_tokens=False))

            if token_length <= max_tokens:
                return [txt] # If the chunk small enough to fit, then return that chunk from RAG.
            if "\n\n" in txt: # First Paragraph-level then sentence-level
                chunk_parts = txt.split('\n\n')
            elif ". " in txt:
                chunk_parts = txt.split(". ")
           
            else:
                token_ids = chunk_tokenizer.encode(txt, add_special_tokens=False)

                    
                return [
                    chunk_tokenizer.decode(token_ids[i:i+max_tokens])
                                           for i in range(0, len(token_ids), max_tokens - overlap_tokens)]
                

            processed_chunks = []
            for j in chunk_parts: # j is a part of the txt.
                j = j.strip()
                if not j:
                    continue
                sub_chunks = recursive_splitting(j)
                processed_chunks.extend(sub_chunks)
            return processed_chunks

        res_chunks = recursive_splitting(context) # Recursive Splitting to avoid large chunks.
        final_chunks = [] # Add overlap on the chunks
        for i in range(len(res_chunks)):
            chunk = res_chunks[i]
            
            # Overlap Case with previous chunk
            if i > 0:
                prev_tokens = chunk_tokenizer.encode(res_chunks[i-1], add_special_tokens=False)
                overlap = prev_tokens[-overlap_tokens:]
                overlap_txt = chunk_tokenizer.decode(overlap)

                chunk = overlap_txt + " " + chunk 
            token_ids_res = chunk_tokenizer.encode(chunk, add_special_tokens=False)
            if len(token_ids_res) > max_tokens:
                token_ids_res = token_ids_res[:max_tokens]
                chunk = chunk_tokenizer.decode(token_ids_res)
            if len(chunk.split()) >= 3:
                final_chunks.append(chunk.strip())
        
        if not final_chunks:
            return [], None 

        return final_chunks, self.encode(final_chunks)



































