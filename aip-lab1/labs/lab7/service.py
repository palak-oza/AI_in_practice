#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402
from aip.guards import ToolGuard  # noqa: E402
from aip.retrieval import DenseRetriever  # noqa: E402
from labs.lab3.search import build_chunks, load_corpus  # noqa: E402
from labs.lab4.rag import REFUSAL, answer_question  # noqa: E402
from labs.lab6.agent import run_agent  # noqa: E402

app = FastAPI(title="Aurora Policy Assistant", version="1.0")

_PIPELINE = None
_STARTED = time.time()
_EXACT: dict[str, dict] = {}
_SEMANTIC: list[tuple[object, dict]] = []
_REQUESTS: list[dict] = []
SEMANTIC_THRESHOLD = 0.98


@dataclass
class Pipeline:
    retriever: DenseRetriever
    guard: ToolGuard


def pipeline() -> Pipeline:
    """Build and retain the retrieval index and guarded tool configuration."""
    global _PIPELINE
    if _PIPELINE is None:
        chunks = build_chunks(load_corpus(), strategy="markdown", size=400, overlap=150)
        _PIPELINE = Pipeline(
            DenseRetriever(chunks, show_progress=False),
            ToolGuard(max_calls=6, allow={"search_policy", "get_policy_details", "compute_premium"}),
        )
    return _PIPELINE


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


def _cache_key(req: AskRequest) -> str:
    return "|".join((" ".join(req.question.lower().split()), str(req.top_k), req.mode))


def _semantic_hit(question: str) -> dict | None:
    """High-threshold cache only; an embedding failure simply skips the cache."""
    if not _SEMANTIC:
        return None
    try:
        from aip.embed import cosine, embed
        query = embed(question)
        vector, payload = max(_SEMANTIC, key=lambda item: float(cosine(query, item[0])))
        return payload.copy() if float(cosine(query, vector)) >= SEMANTIC_THRESHOLD else None
    except Exception:
        return None


def _answer(req: AskRequest) -> tuple[dict, bool]:
    key = _cache_key(req)
    if key in _EXACT:
        return _EXACT[key].copy(), True
    semantic = _semantic_hit(req.question)
    if semantic is not None:
        return semantic, True
    p = pipeline()
    if req.mode == "tools":
        result = run_agent(req.question, guard=p.guard, layers={1, 2, 3, 4, 5})
        payload = {"answer": result["answer"] or REFUSAL, "refused": not bool(result["answer"]), "citations": []}
    else:
        result = answer_question(req.question, p.retriever, k=max(12, req.top_k), final_k=req.top_k)
        payload = {"answer": result.text, "refused": result.refused,
                   "citations": [{"index": i, "doc_id": hit.doc_id, "excerpt": hit.text[:900]}
                                 for i, hit in enumerate(result.hits, 1)]}
    _EXACT[key] = payload.copy()
    try:
        from aip.embed import embed
        _SEMANTIC.append((embed(req.question), payload.copy()))
    except Exception:
        pass
    return payload, False


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a validated request and expose its cost and trace identifier."""
    t0, trace_id = time.perf_counter(), uuid.uuid4().hex
    try:
        with tracing.trace("http.ask", trace_id=trace_id, question=req.question[:120]) as span:
            payload, cached = _answer(req)
            span.update(cached=cached, refused=payload["refused"])
            response = AskResponse(**payload, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                   cost_usd=round(global_budget().spent_usd, 6), cached=cached, trace_id=trace_id)
            _REQUESTS.append({"latency_ms": response.latency_ms, "cached": cached, "error": None})
            return response
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:                                    # noqa: BLE001
        _REQUESTS.append({"error": type(exc).__name__})
        if any(word in str(exc).lower() for word in ("timeout", "rate limit", "connection", "provider")):
            raise HTTPException(status_code=503, detail="upstream model unavailable", headers={"Retry-After": "30"}) from exc
        raise HTTPException(status_code=500, detail="internal service error") from exc


@app.get("/health")
def health() -> dict:
    """Service readiness and the index/cache state used by this process."""
    p = pipeline()
    return {"status": "ok", "uptime_s": round(time.time() - _STARTED, 1),
            "index_size": len(p.retriever.chunks), "model": "configured MAIN tier",
            "cache": {**cache.stats(), "exact_entries": len(_EXACT), "semantic_entries": len(_SEMANTIC),
                      "semantic_threshold": SEMANTIC_THRESHOLD}}


@app.get("/metrics")
def metrics() -> dict:
    b = global_budget()
    latencies = sorted(row["latency_ms"] for row in _REQUESTS if "latency_ms" in row)
    def pct(p: float) -> float:
        return round(latencies[min(len(latencies) - 1, int((len(latencies) - 1) * p))], 1) if latencies else 0.0
    errors: dict[str, int] = {}
    for row in _REQUESTS:
        if row.get("error"):
            errors[row["error"]] = errors.get(row["error"], 0) + 1
    n = len(_REQUESTS)
    return {**b.as_dict(), "cost_per_query_usd": round(b.spent_usd / n, 6) if n else 0.0,
            "cache_hit_rate": round(sum(bool(row.get("cached")) for row in _REQUESTS) / n, 4) if n else 0.0,
            "latency_p50_ms": pct(.5), "latency_p95_ms": pct(.95), "latency_p99_ms": pct(.99),
            "error_rate_by_type": errors, "tool_call_count": 0}


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """SSE endpoint: prose comes first; citations and validation come last."""
    from sse_starlette.sse import EventSourceResponse
    import asyncio
    import json

    async def events():
        trace_id, t0 = uuid.uuid4().hex, time.perf_counter()
        try:
            payload, cached = await asyncio.to_thread(_answer, req)
            for token in payload["answer"].split():
                yield {"event": "token", "data": token + " "}
            yield {"event": "citations", "data": json.dumps(payload["citations"])}
            yield {"event": "validation", "data": json.dumps({"valid": True, "cached": cached, "trace_id": trace_id, "total_ms": round((time.perf_counter()-t0)*1000, 2)})}
        except BudgetExceeded as exc:
            yield {"event": "error", "data": json.dumps({"status": 429, "detail": str(exc)})}
        except Exception:
            yield {"event": "error", "data": json.dumps({"status": 503, "retry_after": 30})}
    return EventSourceResponse(events())
