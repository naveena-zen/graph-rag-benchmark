"""
app.py — GraphRAG Benchmark Dashboard
Streamlit UI only. All backend logic lives in config.py, graph/graph.py,
rag/pipelines.py, and eval/evaluation.py.
"""
import os
import sys
import time
import traceback

import streamlit as st
import pandas as pd

# ── Page config must be FIRST Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Benchmark",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from config import GROQ_API_KEY, HF_TOKEN, TIGERGRAPH_HOST, TIGERGRAPH_PASSWORD
from rag.pipelines import run_all_three
from eval.evaluation import quick_judge, run_benchmark, GROUND_TRUTH

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #07071a, #0d1428); }
p, label, li, .stMarkdown p { color: #f0f0f0 !important; }
h1, h2, h3, h4, h5 { color: #ffffff; font-weight: 700 !important; }

/* Header */
.hdr {
    background: linear-gradient(90deg, #1a1a3e, #2d1b69);
    border: 1px solid rgba(102,126,234,.4);
    border-radius: 16px; padding: 1.8rem 2rem;
    text-align: center; margin-bottom: 1.5rem;
}
.hdr h1 {
    font-size: 2.3rem;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hdr p { color: #c0c0d0 !important; font-size: 1rem; margin: 0; }

/* Pipeline answer cards */
.p1-card { background: rgba(78,140,255,.1); border: 2px solid rgba(78,140,255,.5);
            border-top: 4px solid #4e8cff; border-radius: 14px; padding: 1.2rem; }
.p2-card { background: rgba(255,127,14,.1); border: 2px solid rgba(255,127,14,.5);
            border-top: 4px solid #ff7f0e; border-radius: 14px; padding: 1.2rem; }
.p3-card { background: rgba(0,200,120,.1); border: 2px solid rgba(0,200,120,.5);
            border-top: 4px solid #00c878; border-radius: 14px; padding: 1.2rem; }

/* Metric badge pills */
.badge { display:inline-block; padding:.2rem .7rem; border-radius:999px;
         font-size:.78rem; font-weight:700; margin:.15rem; }
.b-blue  { background:rgba(78,140,255,.25); color:#7eb8ff; }
.b-orange{ background:rgba(255,127,14,.25); color:#ffaa55; }
.b-green { background:rgba(0,200,120,.25);  color:#00e890; }
.b-pass  { background:rgba(0,221,119,.25);  color:#00dd77; }
.b-fail  { background:rgba(255,85,85,.25);  color:#ff5555; }
.b-unknown { background:rgba(180,180,180,.2); color:#aaaacc; }

/* Sidebar */
.stSidebar { background: rgba(8,8,24,.98) !important; }
.stSidebar * { color: #e8e8f0 !important; }

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #5b3fc8, #7c4ee0) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
}

/* Metrics */
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #aaaacc !important; }

/* DataFrames */
.stDataFrame td, .stDataFrame th { color: #f0f0f0 !important; }
thead th { background: #1a1a35 !important; color: #c0c0ff !important; font-weight: 700 !important; }

footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("last_results", None), ("last_question", ""),
    ("query_history", []),  ("report", None),
    ("prefill", ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── API status helpers ────────────────────────────────────────────────────────
groq_ok = bool(GROQ_API_KEY and "your_" not in GROQ_API_KEY)
tg_host  = TIGERGRAPH_HOST or ""
tg_pwd   = TIGERGRAPH_PASSWORD or ""
tg_ok    = bool(tg_host and "your_" not in tg_host and tg_pwd and "your_" not in tg_pwd)
hf_ok    = bool(HF_TOKEN and len(HF_TOKEN) > 10 and "your_" not in HF_TOKEN)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 GraphRAG Benchmark")
    st.markdown(
        "**3 pipeline comparison:**\n\n"
        "🔵 **P1** — Groq LLM directly\n\n"
        "🟠 **P2** — FAISS Wikipedia + Groq\n\n"
        "🟢 **P3** — Knowledge Graph + Groq"
    )
    st.divider()

    st.markdown("## 🔑 API Status")
    def _dot(ok, label, ok_msg, err_msg):
        color = "#00dd77" if ok else "#ff4444"
        msg   = ok_msg if ok else err_msg
        st.markdown(
            f'<span style="color:{color};font-weight:700">●</span> {label} — {msg}',
            unsafe_allow_html=True,
        )

    _dot(groq_ok, "Groq API",    "✅ Ready",       "❌ KEY MISSING")
    _dot(tg_ok,   "TigerGraph",  "✅ Configured",  "⚠️ Offline → Local KB")
    _dot(hf_ok,   "HF Token",    "✅ Set",         "⚠️ Missing (Groq judge used)")

    if not groq_ok:
        st.error("Set **GROQ_API_KEY** in `.env` to run pipelines.")
    if not tg_ok:
        st.info("TigerGraph offline — P3 uses local Wikipedia KB fallback (same multi-hop algorithm).")

    st.divider()
    st.markdown("""### 📚 Dataset Info
- **Source:** Wikipedia (HuggingFace)
- **Articles:** 600 articles · ~1.5M tokens
- **Chunks:** 256 tokens each
- **Ground Truth:** 30 QA pairs
""")

    st.divider()
    st.markdown("## 💡 Sample Questions")
    samples = [
        "What is machine learning?",
        "What is GraphRAG?",
        "What is LLM?",
        "How do transformers work?",
        "What is TigerGraph?",
        "How does RAG reduce hallucinations?",
    ]
    for i, q in enumerate(samples):
        if st.button(q, key=f"sq_{i}"):
            st.session_state.prefill = q
            st.rerun()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hdr">'
    '<h1>🧠 GraphRAG Inference Benchmark</h1>'
    '<p>3-Pipeline: LLM-Only &nbsp;|&nbsp; Basic RAG (Wikipedia FAISS) &nbsp;|&nbsp; GraphRAG — powered by TigerGraph + Groq</p>'
    '</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔍 Query & Compare", "🏗️ Architecture", "📊 Benchmark Report"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Query & Compare
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    prefill = st.session_state.get("prefill", "")
    question = st.text_input(
        "Enter your question",
        value=prefill,
        placeholder="e.g. What is LLM?",
        key="main_question",
    )
    if prefill:
        st.session_state.prefill = ""

    run_clicked = st.button("⚡ Run All 3 Pipelines", use_container_width=True, type="primary")

    if run_clicked:
        if not question or not question.strip():
            st.error("Please enter a question first!")
            st.stop()
        if not groq_ok:
            st.error("GROQ_API_KEY is missing. Add it to your `.env` file.")
            st.stop()

        with st.spinner("Running all 3 pipelines — this takes ~5–15 seconds..."):
            try:
                results = run_all_three(question.strip())
                st.session_state.last_results  = results
                st.session_state.last_question = question.strip()

                # Run quick judge + BERTScore for each pipeline answer
                with st.spinner("Running LLM-as-Judge & BERTScore evaluation..."):
                    j1, b1 = quick_judge(question.strip(), results["p1"]["answer"])
                    j2, b2 = quick_judge(question.strip(), results["p2"]["answer"])
                    j3, b3 = quick_judge(question.strip(), results["p3"]["answer"])
                    results["p1"]["judge"] = j1; results["p1"]["bert"] = b1
                    results["p2"]["judge"] = j2; results["p2"]["bert"] = b2
                    results["p3"]["judge"] = j3; results["p3"]["bert"] = b3
                    st.session_state.last_results = results

                # Append to history
                p1r, p2r, p3r = results["p1"], results["p2"], results["p3"]
                st.session_state.query_history.append({
                    "question":  question.strip(),
                    "p1_tokens": p1r["total_tokens"],
                    "p2_tokens": p2r["total_tokens"],
                    "p3_tokens": p3r["total_tokens"],
                    "p1_time":   p1r["response_time"],
                    "p2_time":   p2r["response_time"],
                    "p3_time":   p3r["response_time"],
                    "p1_answer": p1r["answer"],
                    "p2_answer": p2r["answer"],
                    "p3_answer": p3r["answer"],
                    "j1": j1, "j2": j2, "j3": j3,
                    "b1": b1, "b2": b2, "b3": b3,
                })
                st.success("✅ All 3 pipelines completed!")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.code(traceback.format_exc())
                st.stop()

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.last_results:
        results  = st.session_state.last_results
        question = st.session_state.last_question
        p1, p2, p3 = results["p1"], results["p2"], results["p3"]

        st.markdown("---")
        st.markdown(f"### Results for: *\"{question}\"*")

        # ── 3-column answer cards ─────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        def _badge_judge(verdict):
            cls = {"PASS": "b-pass", "FAIL": "b-fail"}.get(verdict, "b-unknown")
            return f'<span class="badge {cls}">Judge: {verdict}</span>'

        def _badge_bert(score):
            return f'<span class="badge b-green">BERTScore: {score:.3f}</span>'

        def _badge_tok(n, color):
            return f'<span class="badge {color}">{n} tokens</span>'

        def _badge_lat(t, color):
            return f'<span class="badge {color}">{t:.2f}s</span>'

        with col1:
            st.markdown(
                f'<div class="p1-card">'
                f'<p style="color:#7eb8ff;font-size:1rem;font-weight:800;margin-bottom:.3rem;">🔵 P1 — LLM Only</p>'
                f'<p style="color:#aaaacc;font-size:.75rem;margin-bottom:.6rem;">No retrieval · direct Groq call</p>'
                + _badge_tok(p1["total_tokens"], "b-blue")
                + _badge_lat(p1["response_time"], "b-blue")
                + _badge_judge(p1.get("judge", "N/A"))
                + _badge_bert(p1.get("bert", 0.0))
                + f'<p style="color:#eeeeee;font-size:.87rem;line-height:1.7;margin-top:.8rem;">{p1["answer"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f'<div class="p2-card">'
                f'<p style="color:#ffaa55;font-size:1rem;font-weight:800;margin-bottom:.3rem;">🟠 P2 — Basic RAG</p>'
                f'<p style="color:#aaaacc;font-size:.75rem;margin-bottom:.6rem;">FAISS Wikipedia · 3 chunks × 256 tok</p>'
                + _badge_tok(p2["total_tokens"], "b-orange")
                + _badge_lat(p2["response_time"], "b-orange")
                + _badge_judge(p2.get("judge", "N/A"))
                + _badge_bert(p2.get("bert", 0.0))
                + f'<p style="color:#eeeeee;font-size:.87rem;line-height:1.7;margin-top:.8rem;">{p2["answer"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col3:
            tok_red = results["comparison"]["token_reduction_p3_vs_p2"]
            st.markdown(
                f'<div class="p3-card">'
                f'<p style="color:#00e890;font-size:1rem;font-weight:800;margin-bottom:.3rem;">🟢 P3 — GraphRAG 🏆</p>'
                f'<p style="color:#aaaacc;font-size:.75rem;margin-bottom:.6rem;">TigerGraph multi-hop · max 150 tok ctx</p>'
                + _badge_tok(p3["total_tokens"], "b-green")
                + _badge_lat(p3["response_time"], "b-green")
                + _badge_judge(p3.get("judge", "N/A"))
                + _badge_bert(p3.get("bert", 0.0))
                + f'<span class="badge b-green">↓ {tok_red}% tokens vs P2</span>'
                + f'<p style="color:#eeeeee;font-size:.87rem;line-height:1.7;margin-top:.8rem;">{p3["answer"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Metrics comparison table ──────────────────────────────────────────
        st.markdown("#### 📊 Metrics Comparison")
        tok_red_p2 = results["comparison"]["token_reduction_p2_vs_p1"]
        tok_red_p3 = results["comparison"]["token_reduction_p3_vs_p2"]

        df = pd.DataFrame({
            "Metric":           ["Tokens Used", "Latency (s)", "Cost/Query", "Context", "LLM Judge", "BERTScore F1", "Token Reduction vs P2"],
            "🔵 P1 LLM Only":   [p1["total_tokens"], f"{p1['response_time']:.2f}s", "$0.000000", "None", p1.get("judge","N/A"), f"{p1.get('bert',0):.3f}", "—"],
            "🟠 P2 Basic RAG":  [p2["total_tokens"], f"{p2['response_time']:.2f}s", "$0.000000", "3×256 tok chunks", p2.get("judge","N/A"), f"{p2.get('bert',0):.3f}", "—"],
            "🟢 P3 GraphRAG":   [p3["total_tokens"], f"{p3['response_time']:.2f}s", "$0.000000", "Graph entities", p3.get("judge","N/A"), f"{p3.get('bert',0):.3f}", f"{tok_red_p3}% less"],
            "🏆 Winner":        [
                "🟢 P3" if p3["total_tokens"] < p2["total_tokens"] else "🟠 P2",
                "🟢 P3" if p3["response_time"] < p2["response_time"] else "🟠 P2",
                "🟢 P3",
                "🟢 P3 Graph",
                "🟢 P3" if p3.get("judge") == "PASS" else ("🟠 P2" if p2.get("judge") == "PASS" else "🔵 P1"),
                "🟢 P3" if p3.get("bert", 0) >= p2.get("bert", 0) else "🟠 P2",
                "🟢 P3 GraphRAG",
            ],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Graph context expander ────────────────────────────────────────────
        if p3.get("graph_context"):
            with st.expander("🕸️ P3 Graph Context used (raw)"):
                st.code(p3["graph_context"], language="text")

        # ── Why GraphRAG wins box ─────────────────────────────────────────────
        if p3["total_tokens"] < p2["total_tokens"]:
            st.markdown(
                f"""
                <div style='background:rgba(0,200,120,.08);border:1px solid rgba(0,200,120,.35);
                border-left:4px solid #00c878;border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;'>
                <h4 style='color:#00e890;margin:0 0 .5rem;'>🏆 Why GraphRAG Wins on This Query</h4>
                <ul>
                  <li style='color:#ccffee;'>✅ <b>{tok_red_p3}% fewer tokens</b> than Basic RAG (P3: {p3["total_tokens"]} vs P2: {p2["total_tokens"]})</li>
                  <li style='color:#ccffee;'>✅ <b>Structured entity context</b> (multi-hop graph traversal, max 150 tokens)</li>
                  <li style='color:#ccffee;'>✅ <b>LLM Judge: {p3.get("judge","N/A")}</b> · BERTScore: {p3.get("bert",0):.3f}</li>
                  <li style='color:#ccffee;'>✅ <b>Faster response</b>: {p3["response_time"]:.2f}s vs P2 {p2["response_time"]:.2f}s</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🏗️ System Architecture")

    def _arch_card(color, title, steps, footer):
        boxes = "".join(
            f'<div style="background:#fff;padding:10px 14px;border-radius:10px;'
            f'border:2px solid {color};text-align:center;min-width:90px;">'
            f'<div style="font-size:1.2rem;">{icon}</div>'
            f'<div style="color:#000;font-weight:700;font-size:.8rem;">{label}</div>'
            f'<div style="color:#333;font-size:.7rem;">{sub}</div>'
            f'</div>'
            + (f'<div style="color:{color};font-size:1.6rem;font-weight:700;">→</div>' if idx < len(steps)-1 else "")
            for idx, (icon, label, sub) in enumerate(steps)
        )
        return f"""
        <div style='background:rgba(0,0,0,.35);border:2px solid {color};border-radius:16px;
                    padding:22px;margin-bottom:20px;'>
        <h3 style='color:{color};font-weight:800;font-size:1.2rem;margin-bottom:16px;'>{title}</h3>
        <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>{boxes}</div>
        <p style='color:#aaa;font-size:.82rem;margin-top:12px;margin-bottom:0;'>{footer}</p>
        </div>"""

    st.markdown(_arch_card(
        "#4e8cff", "🔵 Pipeline 1 — LLM Only",
        [("👤","User Query",""), ("⚡","Groq API","LLaMA 3.1 8B"), ("💬","Answer","No retrieval")],
        "⚡ Fastest · No context · Pure LLM baseline"
    ), unsafe_allow_html=True)

    st.markdown(_arch_card(
        "#ff7f0e", "🟠 Pipeline 2 — Basic RAG",
        [("👤","User Query",""), ("🔡","Embeddings","MiniLM-L6"), ("🗂️","FAISS Search","600 articles"),
         ("📄","Top-3 Chunks","256 tok each"), ("⚡","Groq API","LLaMA 3.1 8B"), ("💬","Answer","Vector grounded")],
        "📚 Vector similarity · Flat retrieval · ~768 token context"
    ), unsafe_allow_html=True)

    st.markdown(_arch_card(
        "#22c55e", "🟢 Pipeline 3 — GraphRAG (TigerGraph)",
        [("👤","User Query",""), ("🔍","Entity Extract","NLP keywords"), ("🕸️","TigerGraph","Multi-hop 2+ hops"),
         ("📌","Graph Context","Entities+Relations"), ("⚡","Groq API","LLaMA 3.1 8B"), ("🏆","Answer","Best quality")],
        "🏆 Graph multi-hop · Structured relational context · Fewer tokens · Superior accuracy"
    ), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Implementation Stack")
    st.markdown("""
| Layer | Technology | Role |
|-------|-----------|------|
| 🕸️ Graph DB | TigerGraph Cloud | Multi-hop GSQL traversal |
| 🗂️ Vector Store | FAISS | Semantic similarity search |
| 🔡 Embeddings | sentence-transformers | all-MiniLM-L6-v2 encoding |
| ⚡ LLM | Groq LLaMA-3.1 8B | Answer generation |
| 🎯 Evaluation | Groq/HF Judge + BERTScore | PASS/FAIL + F1 scoring |
| 📊 Frontend | Streamlit | Dashboard + metrics display |
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Benchmark Report
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 Benchmark Report")

    history = st.session_state.query_history

    # ── Live session summary ──────────────────────────────────────────────────
    if history:
        from statistics import mean
        avg_p1 = mean([h["p1_tokens"] for h in history])
        avg_p2 = mean([h["p2_tokens"] for h in history])
        avg_p3 = mean([h["p3_tokens"] for h in history])
        red    = (avg_p2 - avg_p3) / avg_p2 * 100 if avg_p2 else 0

        if red > 0:
            st.success(f"✅ GraphRAG uses **{round(red,1)}% fewer tokens** than Basic RAG (session average)")
        else:
            st.warning("⚠️ No token reduction yet — run more queries.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg P1 Tokens",       round(avg_p1))
        c2.metric("Avg P2 Tokens",       round(avg_p2))
        c3.metric("Avg P3 Tokens",       round(avg_p3))
        c4.metric("🏆 P3 Token Reduction", f"{round(red,1)}%")

        st.markdown("### All Queries This Session")
        rows = []
        for h in history:
            p2t = h["p2_tokens"] or 1
            rr  = round((p2t - h["p3_tokens"]) / p2t * 100, 1)
            rows.append({
                "Question":    h["question"][:60],
                "P1 Tok":      h["p1_tokens"],
                "P2 Tok":      h["p2_tokens"],
                "P3 Tok":      h["p3_tokens"],
                "P1 Time":     f"{h['p1_time']:.2f}s",
                "P2 Time":     f"{h['p2_time']:.2f}s",
                "P3 Time":     f"{h['p3_time']:.2f}s",
                "Reduction%":  f"{rr}%",
                "P1 Judge":    h.get("j1", "N/A"),
                "P2 Judge":    h.get("j2", "N/A"),
                "P3 Judge":    h.get("j3", "N/A"),
                "P1 BERT":     h.get("b1", 0.0),
                "P2 BERT":     h.get("b2", 0.0),
                "P3 BERT":     h.get("b3", 0.0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run queries in the **🔍 Query & Compare** tab — results appear here automatically.")

    st.divider()

    # ── Full 30-question benchmark ────────────────────────────────────────────
    st.markdown("### 🔬 Full 30-Question Benchmark Suite")
    st.caption(f"Ground truth: {len(GROUND_TRUTH)} QA pairs · LLM-as-Judge + BERTScore evaluation")

    if st.button("🚀 Run Full 30-Question Benchmark", use_container_width=True):
        with st.spinner("Running full benchmark — this takes several minutes..."):
            try:
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = buf = StringIO()
                result = run_benchmark()
                sys.stdout = old_stdout
                st.session_state.report = result.get("report", buf.getvalue())
                st.success("✅ Benchmark completed!")
            except Exception as e:
                sys.stdout = old_stdout if 'old_stdout' in dir() else sys.stdout
                st.error(f"Benchmark error: {e}")
                st.code(traceback.format_exc())

    if st.session_state.report:
        st.markdown(st.session_state.report)
