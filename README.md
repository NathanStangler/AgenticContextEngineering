# Deep Learning for Dynamic Context Engineering in Agentic Systems

## Context Engine Integration (Claude-style Agent Loops)

The project includes an integration adapter in `integration.py` that exposes
the core context-engineering lifecycle used by agentic systems:

1. `ingest_turn(role, text, metadata=None)`
2. `build_prompt_context(query, token_budget, mode="dynamic_context", top_k=5)`
3. `record_response(response_text, metadata=None)`
4. `get_metrics()`

Supported modes:
- `full_context`
- `rag`
- `dynamic_context`

Agent integration pattern:
1. Ingest incoming user turn.
2. Build prompt context under a token budget.
3. Send returned context to your base model.
4. Record model response.

Run the integration demo:

```bash
python integration.py
```

## Install

```bash
pip install -r requirements.txt
```

## Run Test

```bash
python test.py
```