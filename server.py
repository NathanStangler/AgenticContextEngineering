from threading import Lock
from mcp.server.fastmcp import FastMCP
from rag import RAG
from summarization import Summarization
from token_usage import count_tokens
import uuid

mcp = FastMCP()
sessions = {}
lock = Lock()
summarizer = Summarization()

def get_session(session_id):
    with lock:
        return sessions.get(session_id)

@mcp.tool()
def create_session():
    session_id = uuid.uuid4().hex
    with lock:
        sessions[session_id] = RAG()
    return {"session_id": session_id}

@mcp.tool()
def get_context(session_id, user_prompt, model_name="google-t5/t5-base", top_k=5, max_summary_tokens=120, min_summary_tokens=40):
    if not user_prompt or not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session with id {session_id} not found")
    retrieved = session.retrieve_top_k(query=user_prompt, k=top_k)
    retrieved_text = [item["text"] for item in retrieved]
    summary = summarizer.summarize(retrieved_text, max_tokens=max_summary_tokens, min_tokens=min_summary_tokens)
    session.add_context(user_prompt)

    return {
        "retrieved_chunks": retrieved,
        "context": summary,
        "retrieved_tokens": count_tokens("\n".join(retrieved_text), model_name=model_name),
        "context_tokens": count_tokens(summary, model_name=model_name)
    }

@mcp.tool()
def insert_response(session_id, agent_response, model_name="google-t5/t5-base"):
    if not agent_response or not agent_response.strip():
        raise ValueError("agent_response cannot be empty")

    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session with id {session_id} not found")
    session.add_context(agent_response)

    return {
        "response_tokens": count_tokens(agent_response, model_name=model_name)
    }

if __name__ == "__main__":
    mcp.run()