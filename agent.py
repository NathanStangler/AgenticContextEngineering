import asyncio
import json
import os
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class ContextEngineeringClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = f"{server_url.rstrip('/')}/mcp"
        self.session_id = None

    def mcp_url(self):
        base_url = self.server_url.rstrip("/")
        if base_url.endswith("/mcp"):
            return base_url
        return f"{base_url}/mcp"

    async def call_tool_async(self, tool_name, arguments=None):
        async with streamable_http_client(self.server_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments or {})

    def call_tool(self, tool_name, arguments=None):
        result = asyncio.run(self.call_tool_async(tool_name, arguments))
        if result is None:
            return {}
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        content = getattr(result, "content", None)
        if content:
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"text": text}
        if isinstance(result, dict):
            return result
        return {"text": str(result)}
        
    def create_session(self):
        try:
            result = self.call_tool("create_session")
            self.session_id = result.get("session_id")
            print(f"Created session: {self.session_id}")
            return self.session_id
        except Exception as e:
            print(f"Failed to create session: {e}")
            raise
    
    def get_context(self, user_prompt, top_k=5, max_summary_tokens=120, min_summary_tokens=40, token_budget=600, use_reranker=True, mode="dynamic_context"):
        try:
            result = self.call_tool("get_context", {
                    "session_id": self.session_id,
                    "user_prompt": user_prompt,
                    "top_k": top_k,
                    "max_summary_tokens": max_summary_tokens,
                    "min_summary_tokens": min_summary_tokens,
                    "token_budget": token_budget,
                    "use_reranker": use_reranker,
                    "mode": mode,
                    "model_name": "facebook/bart-large-cnn",
                }
            )
            return result
        except Exception as e:
            print(f"Failed to get context: {e}")
            raise
    
    def insert_response(self, agent_response):
        try:
            result = self.call_tool("insert_response", {
                    "session_id": self.session_id,
                    "agent_response": agent_response,
                    "model_name": "facebook/bart-large-cnn",
                }
            )
            return result
        except Exception as e:
            print(f"Failed to insert response: {e}")
            raise

class ContextEngineeringAgent:
    def __init__(self, server_url="http://localhost:8000", api_key=None, model_name="gemini-2.5-flash", temperature=0.7, use_reranker=True, context_mode="dynamic_context", max_summary_tokens=120, min_summary_tokens=40, token_budget=600, top_k=5):
        self.client = ContextEngineeringClient(server_url)
        self.use_reranker = use_reranker
        self.context_mode = context_mode
        self.max_summary_tokens = max_summary_tokens
        self.min_summary_tokens = min_summary_tokens
        self.token_budget = token_budget
        self.top_k = top_k
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found")
        
        self.model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature, convert_system_message_to_human=True)
        self.client.create_session()

        @tool("query_memory", description="Query the agent's memory for relevant context based on a user prompt")
        def query_memory(query):
            return self._query_memory(query)

        @tool("save_to_memory", description="Save important information to memory for future context retrieval")
        def save_to_memory(content):
            return self._save_to_memory(content)

        self.query_memory_tool = query_memory
        self.save_to_memory_tool = save_to_memory
        self.agent = create_agent(tools=[self.query_memory_tool, self.save_to_memory_tool], model=self.model)

    def _build_prompt(self, user_query, context):
        if context:
            return (
                "Use the context when it is relevant to the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_query}\n\n"
                "Answer clearly and concisely."
            )

        return f"Question: {user_query}\n\nAnswer clearly and concisely."

    def _query_memory(self, query):
        try:
            result = self.client.get_context(
                user_prompt=query,
                top_k=self.top_k,
                max_summary_tokens=self.max_summary_tokens,
                min_summary_tokens=self.min_summary_tokens,
                token_budget=self.token_budget,
                use_reranker=self.use_reranker,
                mode=self.context_mode
            )
            return result.get("context", "")
        except Exception as e:
            return f"Error querying memory: {e}"

    def _save_to_memory(self, content):
        try:
            result = self.client.insert_response(content)
            response_tokens = result.get("response_tokens", 0)
            return f"Successfully saved to memory ({response_tokens} tokens)"
        except Exception as e:
            return f"Error saving to memory: {e}"
    
    def run(self, user_query):
        try:
            context = self._query_memory(user_query)
            prompt = self._build_prompt(user_query, context)
            response = self.model.invoke([HumanMessage(content=prompt)])
            response_text = response.content if isinstance(response, AIMessage) else getattr(response, "content", "")
            if response_text:
                self._save_to_memory(response_text)
            return {"response": response_text, "success": True}
        except Exception as e:
            return {"response": f"Error: {e}", "success": False}

if __name__ == "__main__":
    agent = ContextEngineeringAgent(server_url="http://localhost:8000", use_reranker=True, context_mode="dynamic_context", max_summary_tokens=120)
    print("Context Engineering Chatbot (type 'exit' or 'quit' to stop)")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            break

        result = agent.run(user_input)
        print("\n" + "=" * 80)
        print("AGENT RESPONSE:")
        print("=" * 80)
        print(result["response"])