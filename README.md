# Deep Learning for Dynamic Context Engineering in Agentic Systems

## Overview
This project builds a dynamic context engine for long-running conversations. Given a new user prompt and an accumulated history, the system retrieves, re-ranks, and summarizes prior context under a token budget.

Main components:
- FAISS based retriever with sentence-transformer embeddings.
- Learned cross-attention re-ranker for query conditioned chunk importance.
- Summarization pipeline for token efficient context packaging.
- MCP server exposing context tools for agent integration.
- Benchmarking and training utilities.

## Setup

```bash
pip install -r requirements.txt
```

## Core Context Engine
The integration adapter in integration.py exposes the standard agent loop lifecycle:

1. `ingest_turn(role, text, metadata=None)`
2. `build_prompt_context(query, token_budget, mode="dynamic_context", top_k=5, use_reranker=True)`
3. `record_response(response_text, metadata=None)`
4. `get_metrics()`

Supported modes:
- `full_context`
- `rag`
- `dynamic_context`

Run the demo:

```bash
python integration.py
```

## MCP Server
Start the server:

```bash
python server.py
```

## Agent Demo (LangChain + Gemini)
The demo agent calls the MCP server for memory.

```bash
set GOOGLE_API_KEY=your_key_here
python agent.py
```

## Benchmarking
Run the benchmark on the processed dataset:

```bash
python benchmark.py --limit 20 --token-budget 600
```

The benchmark reports NDCG@K, Recall@K, and average context tokens for:
- `rag`
- `dynamic_context`
- `rag_rerank`
- `dynamic_context_rerank`

## Reranker Training
Train the cross-attention re-ranker:

```bash
python train/model_train.py
```

## Dataset Generation
Regenerate the synthetic dataset:

```bash
python data/raw/generate_data.py
```