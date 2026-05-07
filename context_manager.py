from time import perf_counter

from rag import RAG
from reranker import LearnedChunkReranker
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
			"rerank_time_ms": 0.0,
			"context_build_time_ms": 0.0,
			"last_mode": None,
			"last_reranker": None,
			"history": [],
		}


class ContextManager:
	VALID_MODES = {"full_context", "rag", "dynamic_context"}

	def __init__(self, embedding_model = "sentence-transformers/all-MiniLM-L6-v2", summarization_model = "facebook/bart-large-cnn", reranker_weights = "train/model.pth", chunk_max_tokens = 200, overlap_percentage = 0.2):
		self.embedding_model = embedding_model
		self.summarization_model = summarization_model
		self.summarizer = Summarization(model_name=summarization_model)
		self.reranker = LearnedChunkReranker(weights_path=reranker_weights)
		self.chunk_max_tokens = chunk_max_tokens
		self.overlap_percentage = overlap_percentage
		self.session = self._new_session()

	def _new_session(self):
		return SessionState(
			rag=RAG(
				embedding_model=self.embedding_model,
				chunk_max_tokens=self.chunk_max_tokens,
				overlap_percentage=self.overlap_percentage,
			),
		)

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
		max_new = max(20, min(120, int(token_budget * 0.25)))
		min_new = max(10, int(max_new * 0.5))
		return {"min_new_tokens": min_new, "max_new_tokens": max_new}

	def _rerank_chunks(self, query, retrieved, use_reranker):
		selected_chunks = [item["text"] for item in retrieved]
		sim_scores = [item.get("score", 0.0) for item in retrieved]

		if not use_reranker or not self.reranker.available or not selected_chunks:
			retrieved_chunks = [
				{"text": selected_chunks[i], "score": float(sim_scores[i]), "rerank_score": None}
				for i in range(len(selected_chunks))
			]
			return {
				"order": list(range(len(selected_chunks))),
				"rerank_scores": None,
				"retrieved_chunks": retrieved_chunks,
				"reranked_chunks": retrieved_chunks,
			}

		r_start = perf_counter()
		rerank_scores = self.reranker.score(query, selected_chunks)
		r_time_ms = (perf_counter() - r_start) * 1000

		order = sorted(range(len(selected_chunks)), key=lambda idx: rerank_scores[idx], reverse=True)
		retrieved_chunks = [
			{"text": selected_chunks[i], "score": float(sim_scores[i]), "rerank_score": float(rerank_scores[i])}
			for i in range(len(selected_chunks))
		]
		reranked_chunks = [
			{"text": selected_chunks[i], "score": float(sim_scores[i]), "rerank_score": float(rerank_scores[i])}
			for i in order
		]
		return {
			"order": order,
			"rerank_scores": rerank_scores,
			"retrieved_chunks": retrieved_chunks,
			"reranked_chunks": reranked_chunks,
			"rerank_time_ms": r_time_ms,
		}

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

	def build_prompt_context(self, query, token_budget, mode = "dynamic_context", top_k = 5, use_reranker = True, summary_min_tokens = None, summary_max_tokens = None):
		if mode not in self.VALID_MODES:
			raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")

		session = self._get_session()
		started = perf_counter()
		rerank_time_ms = 0.0
		reranked_chunks = []
		retrieved_chunks = []
		reranker_used = False

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
			rerank_pkg = self._rerank_chunks(query, retrieved, use_reranker=use_reranker)
			retrieved_chunks = rerank_pkg["retrieved_chunks"]
			reranked_chunks = rerank_pkg["reranked_chunks"]
			r_order = rerank_pkg["order"]
			selected_chunks = [retrieved_chunks[i]["text"] for i in r_order]
			rerank_time_ms = rerank_pkg.get("rerank_time_ms", 0.0)
			reranker_used = use_reranker and self.reranker.available

			if mode == "rag":
				context_text = "\n".join(selected_chunks)
				summary = ""
				summarization_ms = 0.0
			else:
				sum_start = perf_counter()
				if summary_min_tokens is not None or summary_max_tokens is not None:
					min_new = summary_min_tokens if summary_min_tokens is not None else 10
					max_new = summary_max_tokens if summary_max_tokens is not None else max(min_new, 40)
					limits = {"min_new_tokens": min_new, "max_new_tokens": max_new}
				else:
					limits = self._allocate_summary_tokens(token_budget)
				summary = self.summarizer.summarize(
					selected_chunks,
					max_tokens=limits["max_new_tokens"],
					min_tokens=limits["min_new_tokens"],
				)
				summarization_ms = (perf_counter() - sum_start) * 1000

				context_text = "\n".join(
					[
						"Summary:",
						summary,
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
			"rerank_time_ms": rerank_time_ms,
			"context_build_time_ms": elapsed_ms,
		}
		session.metrics["token_context"] += context_tokens
		session.metrics["retrieval_time_ms"] += retrieval_ms
		session.metrics["summarization_time_ms"] += summarization_ms
		session.metrics["rerank_time_ms"] += rerank_time_ms
		session.metrics["context_build_time_ms"] += elapsed_ms
		session.metrics["last_mode"] = mode
		session.metrics["last_reranker"] = reranker_used
		session.metrics["history"].append(history_row)

		return {
			"mode": mode,
			"query": query,
			"context": clipped,
			"context_tokens": context_tokens,
			"token_budget": token_budget,
			"selected_chunks": selected_chunks,
			"retrieved_chunks": retrieved_chunks,
			"reranked_chunks": reranked_chunks,
			"reranker_used": reranker_used,
			"summary": summary,
			"timing_ms": {
				"retrieval": retrieval_ms,
				"summarization": summarization_ms,
				"rerank": rerank_time_ms,
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
		min_summary = payload.get("min_summary_tokens")
		max_summary = payload.get("max_summary_tokens")
		min_summary = int(min_summary) if min_summary is not None else None
		max_summary = int(max_summary) if max_summary is not None else None
		return self.build_prompt_context(
			query=payload["query"],
			token_budget=int(payload.get("token_budget", 600)),
			mode=payload.get("mode", "dynamic_context"),
			top_k=int(payload.get("top_k", 5)),
			use_reranker=bool(payload.get("use_reranker", True)),
			summary_min_tokens=min_summary,
			summary_max_tokens=max_summary,
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
