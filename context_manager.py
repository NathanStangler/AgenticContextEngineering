from time import perf_counter

from rag import RAG
from summarization import Summarization
from token_usage import count_tokens


class Turn:
	def __init__(self, role, text, metadata=None):
		self.role = role
		self.text = text
		self.metadata = metadata or {}


class SessionState:
	def __init__(self, rag=None):
		self.rag = rag or RAG()
		self.turns = []
		self.metrics = {
			"turn_count": 0,
			"token_in": 0,
			"token_context": 0,
			"retrieval_time_ms": 0.0,
			"summarization_time_ms": 0.0,
			"context_build_time_ms": 0.0,
			"last_mode": None,
			"history": [],
		}


class ContextManager:
	VALID_MODES = {"full_context", "rag", "dynamic_context"}

	def __init__(self, embedding_model = "sentence-transformers/all-MiniLM-L6-v2", summarization_model = "google-t5/t5-base"):
		self.embedding_model = embedding_model
		self.summarization_model = summarization_model
		self.summarizer = Summarization(model_name=summarization_model)
		self.session = self._new_session()

	def _new_session(self):
		return SessionState(rag=RAG(embedding_model=self.embedding_model))

	def _get_session(self):
		return self.session

	@staticmethod
	def _join_turns(turns):
		return "\n".join(f"{t.role}: {t.text}" for t in turns if t.text.strip())

	@staticmethod
	def _clip_to_budget(text, token_budget):
		if token_budget <= 0:
			return ""
		if count_tokens(text) <= token_budget:
			return text

		words = text.split()
		lo, hi = 0, len(words)
		best = ""
		while lo <= hi:
			mid = (lo + hi) // 2
			candidate = " ".join(words[:mid])
			if count_tokens(candidate) <= token_budget:
				best = candidate
				lo = mid + 1
			else:
				hi = mid - 1
		return best

	@staticmethod
	def _allocate_summary_tokens(token_budget):
		if token_budget <= 20:
			return {"min_new_tokens": 5, "max_new_tokens": max(10, token_budget // 2)}
		max_new = max(20, min(180, int(token_budget * 0.4)))
		min_new = max(10, int(max_new * 0.5))
		return {"min_new_tokens": min_new, "max_new_tokens": max_new}

	def ingest_turn(self, role, text, metadata = None):
		session = self._get_session()
		payload = text.strip()

		if not payload:
			return {"ok": False, "reason": "empty_text"}

		turn = Turn(role=role, text=payload, metadata=metadata or {})
		session.turns.append(turn)
		session.rag.add_context(f"{role}: {payload}")

		session.metrics["turn_count"] += 1
		session.metrics["token_in"] += count_tokens(payload)
		return {"ok": True, "turn_count": session.metrics["turn_count"]}

	def build_prompt_context(self, query, token_budget, mode = "dynamic_context", top_k = 5):
		if mode not in self.VALID_MODES:
			raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")

		session = self._get_session()
		started = perf_counter()

		if mode == "full_context":
			context_text = self._join_turns(session.turns)
			selected_chunks = []
			summary = ""
			retrieval_ms = 0.0
			summarization_ms = 0.0
		else:
			retrieve_start = perf_counter()
			retrieved = session.rag.retrieve_top_k(query=query, k=top_k)
			retrieval_ms = (perf_counter() - retrieve_start) * 1000
			selected_chunks = [item["text"] for item in retrieved]

			if mode == "rag":
				context_text = "\n".join(selected_chunks)
				summary = ""
				summarization_ms = 0.0
			else:
				sum_start = perf_counter()
				limits = self._allocate_summary_tokens(token_budget)
				summary = self.summarizer.summarize(
					selected_chunks,
					max_tokens=limits["max_new_tokens"],
					min_tokens=limits["min_new_tokens"],
				)
				summarization_ms = (perf_counter() - sum_start) * 1000

				# Keep a compact context package with both compressed and raw signal.
				context_text = "\n".join(
					[
						"Summary:",
						summary,
						"",
						"Key Retrieved Chunks:",
						*[f"- {chunk}" for chunk in selected_chunks[:3]],
					]
				).strip()

		clipped = self._clip_to_budget(context_text, token_budget=token_budget)
		context_tokens = count_tokens(clipped)
		elapsed_ms = (perf_counter() - started) * 1000

		history_row = {
			"mode": mode,
			"query": query,
			"token_budget": token_budget,
			"top_k": top_k,
			"selected_count": len(selected_chunks),
			"context_tokens": context_tokens,
			"retrieval_time_ms": retrieval_ms,
			"summarization_time_ms": summarization_ms,
			"context_build_time_ms": elapsed_ms,
		}
		session.metrics["token_context"] += context_tokens
		session.metrics["retrieval_time_ms"] += retrieval_ms
		session.metrics["summarization_time_ms"] += summarization_ms
		session.metrics["context_build_time_ms"] += elapsed_ms
		session.metrics["last_mode"] = mode
		session.metrics["history"].append(history_row)

		return {
			"mode": mode,
			"query": query,
			"context": clipped,
			"context_tokens": context_tokens,
			"token_budget": token_budget,
			"selected_chunks": selected_chunks,
			"summary": summary,
			"timing_ms": {
				"retrieval": retrieval_ms,
				"summarization": summarization_ms,
				"context_build": elapsed_ms,
			},
		}

	def record_response(self, response_text, metadata = None):
		return self.ingest_turn(
			role="assistant",
			text=response_text,
			metadata=metadata,
		)

	def get_metrics(self):
		session = self._get_session()
		snapshot = dict(session.metrics)
		return snapshot

	def tool_ingest_turn(self, payload):
		return self.ingest_turn(
			role=payload["role"],
			text=payload["text"],
			metadata=payload.get("metadata"),
		)

	def tool_build_prompt_context(self, payload):
		return self.build_prompt_context(
			query=payload["query"],
			token_budget=int(payload.get("token_budget", 600)),
			mode=payload.get("mode", "dynamic_context"),
			top_k=int(payload.get("top_k", 5)),
		)

	def tool_record_response(self, payload):
		return self.record_response(
			response_text=payload["response"],
			metadata=payload.get("metadata"),
		)

	def tool_get_metrics(self, payload):
		return self.get_metrics()


def main():
	cm = ContextManager()

	cm.ingest_turn("user", "I care about reducing token costs.")
	cm.ingest_turn("assistant", "We can use retrieval and summary compression.")
	cm.ingest_turn("user", "My final report is due Friday and I have a 3 PM meeting.")

	pkg = cm.build_prompt_context(
		query="What are my priorities and deadlines?",
		token_budget=120,
		mode="dynamic_context",
		top_k=4,
	)

	print("Context package:")
	print(pkg["context"])
	print("\nMetrics:")
	print(cm.get_metrics())


if __name__ == "__main__":
	main()
