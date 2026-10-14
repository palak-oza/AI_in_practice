#!/usr/bin/env python3
"""Streamlit chat UI for the Lab 7 Aurora Policy Assistant."""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Aurora Policy Assistant", page_icon="A", layout="wide")

API = st.sidebar.text_input("Service URL", "http://localhost:8000").rstrip("/")
stream_answer = st.sidebar.toggle("Stream the answer", value=True)
mode = st.sidebar.radio("Mode", ("rag", "tools"), index=0, help="RAG answers cite policy passages. Tools mode uses the guarded tool agent.")

st.title("Aurora Policy Assistant")
st.caption("Answers come only from Aurora's policy documents. Every claim is cited. When the documents do not cover a question, the assistant says so instead of guessing.")
question = st.text_input("Ask a question", placeholder="How many days do I have to submit a reimbursement claim after discharge?")

if st.button("Ask", type="primary") and question.strip():
    payload = {"question": question.strip(), "mode": mode}
    answer = ""
    citations = []
    trace_id = ""
    cached = False
    cost_usd = 0.0
    latency_ms = 0.0
    first_token_ms = None
    refused = False
    error = None
    answer_area = st.empty()

    try:
        if stream_answer:
            started = time.perf_counter()
            event_name = "message"
            data_lines = []
            with requests.post(f"{API}/ask/stream", json=payload, stream=True, timeout=180) as response:
                response.raise_for_status()
                for raw in response.iter_lines(decode_unicode=True):
                    if raw == "":
                        event_data = "\n".join(data_lines)
                        if event_name == "token":
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - started) * 1000
                            answer += event_data
                            answer_area.markdown(answer)
                        elif event_name == "citations":
                            citations = json.loads(event_data) if event_data else []
                        elif event_name == "validation":
                            validation = json.loads(event_data) if event_data else {}
                            latency_ms = validation.get("total_ms", (time.perf_counter() - started) * 1000)
                            cached = validation.get("cached", False)
                            trace_id = validation.get("trace_id", "")
                            cost_usd = validation.get("cost_usd", 0.0)
                            refused = validation.get("refused", False)
                        elif event_name == "error":
                            details = json.loads(event_data) if event_data else {}
                            raise RuntimeError(details.get("detail", "Streaming request failed"))
                        event_name, data_lines = "message", []
                    elif raw.startswith("event:"):
                        event_name = raw[6:].strip()
                    elif raw.startswith("data:"):
                        data_lines.append(raw[5:].lstrip())
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            if not latency_ms:
                latency_ms = (time.perf_counter() - started) * 1000
        else:
            started = time.perf_counter()
            response = requests.post(f"{API}/ask", json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            answer = result.get("answer", "")
            citations = result.get("citations", [])
            trace_id = result.get("trace_id", "")
            cached = result.get("cached", False)
            cost_usd = result.get("cost_usd", 0.0)
            latency_ms = result.get("latency_ms", (time.perf_counter() - started) * 1000)
            refused = result.get("refused", False)
            answer_area.markdown(answer) if not refused else answer_area.warning(answer)
    except requests.HTTPError as exc:
        body = exc.response.text[:500] if exc.response is not None else str(exc)
        st.error(f"Service returned {exc.response.status_code if exc.response else 'an error'}: {body}")
        st.stop()
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        st.error(f"Could not complete the request: {exc}")
        st.stop()

    if stream_answer and refused:
        answer_area.warning(answer)

    if citations:
        st.subheader("Sources")
        for citation in citations:
            label = f"[{citation.get('index', '?')}] {citation.get('doc_id', 'source')}"
            with st.expander(label):
                st.text(citation.get("excerpt", "Source excerpt unavailable."))
    elif mode == "rag":
        st.info("No source passages were returned for this answer.")
    else:
        st.caption("Tools mode does not return document citations.")

    columns = st.columns(5)
    columns[0].metric("latency", f"{latency_ms:.0f} ms")
    columns[1].metric("first token", f"{first_token_ms:.0f} ms" if first_token_ms is not None else "—")
    columns[2].metric("cost", f"${cost_usd:.5f}")
    columns[3].metric("cached", "yes" if cached else "no")
    columns[4].metric("sources", len(citations))
    st.caption(f"trace: `{trace_id}`")
    st.session_state["last_feedback"] = {"question": question.strip(), "answer": answer, "trace_id": trace_id}

if st.session_state.get("last_feedback"):
    if st.button("👎 This answer was wrong or unhelpful"):
        item = {**st.session_state["last_feedback"], "created_at": time.time()}
        queue = ROOT / "data" / "review_queue.jsonl"
        try:
            queue.parent.mkdir(parents=True, exist_ok=True)
            with queue.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            st.success("Feedback saved for review.")
        except OSError as exc:
            st.error(f"Could not save feedback: {exc}")