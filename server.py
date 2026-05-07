import runtime_settings
from threading import Lock
from mcp.server.fastmcp import FastMCP
from token_usage import count_tokens
import uuid

mcp = FastMCP()
sessions = {}
lock = Lock()

def get_session(session_id):
    with lock:
        return sessions.get(session_id)

@mcp.tool()
def create_session():
    from context_manager import ContextManager

    session_id = uuid.uuid4().hex
    manager = ContextManager()
    with lock:
        sessions[session_id] = manager
    return {"session_id": session_id}

@mcp.tool()
def get_context(session_id, user_prompt, model_name="facebook/bart-large-cnn", top_k=5, max_summary_tokens=120, min_summary_tokens=40, token_budget=600, mode="dynamic_context", use_reranker=True):
    if not user_prompt or not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session with id {session_id} not found")
    context = session.build_prompt_context(
        query=user_prompt,
        token_budget=token_budget,
        mode=mode,
        top_k=top_k,
        use_reranker=use_reranker,
        summary_min_tokens=min_summary_tokens,
        summary_max_tokens=max_summary_tokens,
    )
    session.ingest_turn("user", user_prompt)

    retrieved_text = [item["text"] for item in context.get("retrieved_chunks", [])]
    summary = context.get("summary", "")

    return {
        "retrieved_chunks": context.get("retrieved_chunks", []),
        "reranked_chunks": context.get("reranked_chunks", []),
        "context": context.get("context", ""),
        "summary": summary,
        "reranker_used": context.get("reranker_used", False),
        "retrieved_tokens": count_tokens("\n".join(retrieved_text), model_name=model_name),
        "context_tokens": count_tokens(context.get("context", ""), model_name=model_name),
        "timing_ms": context.get("timing_ms", {}),
        "mode": mode,
        "token_budget": token_budget,
    }

@mcp.tool()
def insert_response(session_id, agent_response, model_name="facebook/bart-large-cnn"):
    if not agent_response or not agent_response.strip():
        raise ValueError("agent_response cannot be empty")

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session with id {session_id} not found")
    session.record_response(agent_response)

    return {
        "response_tokens": count_tokens(agent_response, model_name=model_name)
    }

@mcp.tool()
def get_metrics(session_id):
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session with id {session_id} not found")
    return session.get_metrics()

if __name__ == "__main__":
    mcp.run()
