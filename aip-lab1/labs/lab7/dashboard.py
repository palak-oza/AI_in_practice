#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — operations")

runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

chosen = st.sidebar.multiselect("runs", [p.stem for p in runs],
                                default=[runs[0].stem])
rows = [json.loads(l) for p in runs if p.stem in chosen
        for l in p.open(encoding="utf-8") if l.strip()]
if not rows:
    st.stop()

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"], unit="s")

c = st.columns(5)
c[0].metric("spans", len(df))
c[1].metric("total cost", f"${df.get('cost_usd', pd.Series([0])).fillna(0).sum():.4f}")
llm = df[df["name"] == "llm.call"]
c[2].metric("model calls", len(llm))
if len(llm):
    c[3].metric("cache hit rate", f"{llm.get('cached', pd.Series([False])).fillna(False).mean():.0%}")
c[4].metric("errors", int((df.get("status") == "error").sum()))

st.subheader("Latency by stage")
stage = (df.groupby("name")["duration_ms"]
         .agg(n="count", p50="median",
              p95=lambda s: s.quantile(0.95), total="sum")
         .sort_values("total", ascending=False))
st.dataframe(stage, use_container_width=True)

st.subheader("Cost over time")
if "cost_usd" in df:
    cum = df.sort_values("ts").assign(cum=lambda d: d["cost_usd"].fillna(0).cumsum())
    st.line_chart(cum.set_index("ts")["cum"])

st.subheader("Errors")
errs = df[df.get("status") == "error"]
st.dataframe(errs[["ts", "name", "error"]] if len(errs) else pd.DataFrame(),
             use_container_width=True)

st.subheader("Alert")
ask_spans = df[df["name"].isin(["http.ask", "http.ask.stream"])]
if len(ask_spans) >= 10:
    recent = ask_spans.tail(10)
    baseline = ask_spans.head(max(1, len(ask_spans) - 10))
    recent_refusals = recent.get("refused", pd.Series(False, index=recent.index)).fillna(False).mean()
    baseline_refusals = baseline.get("refused", pd.Series(False, index=baseline.index)).fillna(False).mean()
    if baseline_refusals and recent_refusals >= baseline_refusals * 2:
        st.error("Refusal rate doubled: pause deployments and check corpus freshness, indexing, and retrieval traces.")
    else:
        st.success("Refusal-rate alert is clear. It triggers when the recent rate is double the baseline.")
else:
    st.info("The refusal-rate alert needs ten requests. On trigger: check index freshness and retrieval traces before deploying.")
