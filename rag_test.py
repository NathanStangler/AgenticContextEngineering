from rag import RAG

rag = RAG()

text = """
Python is a high-level programming language known for its simplicity and readability.
It is widely used in machine learning, data science, web development, automation, and AI systems.

In machine learning, Python provides powerful libraries such as NumPy, Pandas, and Scikit-learn.
Deep learning frameworks like PyTorch and TensorFlow are also built around Python.

FAISS (Facebook AI Similarity Search) is a library developed by Meta for efficient similarity search
and clustering of dense vectors. It is commonly used in large-scale retrieval systems and RAG pipelines.

SentenceTransformers is a Python framework built on top of HuggingFace Transformers.
It allows developers to generate semantically meaningful embeddings for sentences, paragraphs,
and documents. These embeddings are used for semantic search, clustering, and retrieval tasks.

In modern AI applications, retrieval-augmented generation (RAG) systems combine vector databases
with language models to improve factual accuracy and reduce hallucinations.

FAISS is often paired with SentenceTransformers to build fast semantic search engines.

Python also plays a key role in backend development using frameworks like FastAPI and Django.
It is widely adopted in production systems for AI-powered applications, APIs, and automation tools.

Many companies use Python to build recommendation systems, chatbots, and search engines.
"""

rag.add_context(text)



results = rag.retrieve_top_k("What types of work are done with python when it comes to data analysis?", k=5)
print("Total chunks:", len(rag.chunks))
print("FAISS index size:", rag.index.ntotal)

for r in results:
    print(r)