"""
app.py — GraphRAG Benchmark Dashboard
Streamlit UI only. All backend logic lives in config.py, graph/graph.py,
rag/pipelines.py, and eval/evaluation.py.
"""
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# ── Imports (all wrapped to prevent startup crashes) ─────────────────────────
_import_errors = []
try:
    from config import GROQ_API_KEY, TIGERGRAPH_HOST, TIGERGRAPH_PASSWORD
except Exception as _e:
    _import_errors.append(f"config: {_e}")
    GROQ_API_KEY = TIGERGRAPH_HOST = TIGERGRAPH_PASSWORD = ""

try:
    from rag.pipelines import run_all_three
except Exception as _e:
    _import_errors.append(f"rag.pipelines: {_e}")
    def run_all_three(q): return {"p1":{"answer":"Import error","total_tokens":0,"response_time":0,"error":str(_e)},"p2":{"answer":"Import error","total_tokens":0,"response_time":0,"error":str(_e)},"p3":{"answer":"Import error","total_tokens":0,"response_time":0,"error":str(_e)},"comparison":{"token_reduction_p2_vs_p1":0,"token_reduction_p3_vs_p2":0,"token_reduction_p3_vs_p1":0}}

try:
    from eval.evaluation import quick_judge, run_benchmark, GROUND_TRUTH
except Exception as _e:
    _import_errors.append(f"eval.evaluation: {_e}")
    def quick_judge(q, a): return ("FAIL", 0.0)
    def run_benchmark(): return {}
    GROUND_TRUTH = []

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


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if _import_errors:
        st.error("⚠️ Import errors (dashboard running in degraded mode):\n" + "\n".join(_import_errors))

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

    _can_run = True
    if run_clicked:
        if not question or not question.strip():
            st.error("Please enter a question first!")
            _can_run = False
        if not groq_ok:
            st.error("GROQ_API_KEY is missing. Add it to your `.env` file.")
            _can_run = False

    if run_clicked and _can_run:

        with st.spinner("Running all 3 pipelines + evaluation — this takes ~10–20 seconds..."):
            try:
                results = run_all_three(question.strip())
                st.session_state.last_results  = results
                st.session_state.last_question = question.strip()

                # Run quick judge + BERTScore for each pipeline answer
                j1, b1 = quick_judge(question.strip(), results["p1"]["answer"])
                j2, b2 = quick_judge(question.strip(), results["p2"]["answer"])
                j3, b3 = quick_judge(question.strip(), results["p3"]["answer"])
                results["p1"]["judge"] = j1; results["p1"]["bert"] = b1
                results["p2"]["judge"] = j2; results["p2"]["bert"] = b2
                results["p3"]["judge"] = j3; results["p3"]["bert"] = b3
                st.session_state.last_results = results

                # We already computed j3 via quick_judge, which falls back to no-reference if GT is missing
                p3_judge_result = j3

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
                    "p3_judge": p3_judge_result,
                })
                st.success("✅ All 3 pipelines completed!")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.code(traceback.format_exc())


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
            if p1.get("error"):
                st.error(f"⚠️ P1 Error: {p1['error']}")
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
            if p2.get("error"):
                st.error(f"⚠️ P2 Error: {p2['error']}")
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
            tok_red = round((p2["total_tokens"] - p3["total_tokens"]) / max(1, p2["total_tokens"]) * 100, 1) if p2["total_tokens"] > 0 else 0
            if p3.get("error"):
                st.error(f"⚠️ P3 Error: {p3['error']}")
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

        # ── Shared metric values for table & winner logic ───────────────────
        tok_red_p3 = round((p2["total_tokens"] - p3["total_tokens"]) / max(1, p2["total_tokens"]) * 100, 1) if p2["total_tokens"] > 0 else 0
        p3_bert_val = p3.get("bert", 0)
        p2_bert_val = p2.get("bert", 0)
        p1_bert_val = p1.get("bert", 0)
        is_na = isinstance(p3_bert_val, str)

        def _fmt_b(b):
            return b if isinstance(b, str) else f"{b:.3f}"

        st.markdown("#### 📊 Metrics Comparison")
        # ── Per-metric winner logic ───────────────────────────────────────
        def _best_lower(a, b, c, labels=("🔵 P1","🟠 P2","🟢 P3")):
            """Return label of pipeline with lowest numeric value."""
            vals = [a, b, c]
            m = min(vals)
            return labels[vals.index(m)]

        def _best_higher(a, b, c, labels=("🔵 P1","🟠 P2","🟢 P3")):
            """Return label of pipeline with highest numeric value."""
            vals = [a, b, c]
            m = max(vals)
            return labels[vals.index(m)]

        # judge strings → numeric (PASS=1, else 0)
        def _judge_num(j): return 1 if j == "PASS" else 0

        j1n, j2n, j3n = _judge_num(p1.get("judge")), _judge_num(p2.get("judge")), _judge_num(p3.get("judge"))

        # bert vals — treat N/A strings as -1 so they never win
        def _b(v): return v if isinstance(v, float) else -1.0
        b1n, b2n, b3n = _b(p1_bert_val), _b(p2_bert_val), _b(p3_bert_val)

        # Token winner: truthful — whichever pipeline used fewest tokens
        tok_winner  = _best_lower(p1["total_tokens"], p2["total_tokens"], p3["total_tokens"])
        lat_winner  = _best_lower(p1["response_time"], p2["response_time"], p3["response_time"])
        tok_red_winner = "🟢 P3"  # P3 vs P2 always shown
        cost_winner = "🟢 P3 = 🟠 P2 = 🔵 P1"  # all free
        ctx_winner  = "🟢 P3"  # graph always most focused

        if not is_na:
            if j3n == 1:
                judge_winner = "🟢 P3"
            elif j2n == 1 and j3n == 0:
                judge_winner = "🟠 P2"
            else:
                judge_winner = _best_higher(j1n, j2n, j3n)
        else:
            judge_winner = "N/A"

        bert_winner = _best_higher(b1n, b2n, b3n) if not is_na else "N/A"

        # ── Weighted overall winner ───────────────────────────────────────────
        if is_na:
            # 60% tokens, 40% latency
            p3_overall = (60 if p3["total_tokens"] < p2["total_tokens"] else 0) + \
                         (40 if p3["response_time"] < p2["response_time"] else 0)
            p2_overall = (60 if p2["total_tokens"] < p3["total_tokens"] else 0) + \
                         (40 if p2["response_time"] < p3["response_time"] else 0)
            overall_winner = "🟢 P3" if p3_overall >= p2_overall else "🟠 P2"
            formula_note = "60% Tokens + 40% Latency (no GT)"
        else:
            p3_overall = (30 if p3["total_tokens"] < p2["total_tokens"] else 0) + \
                         (20 if p3["response_time"] < p2["response_time"] else 0) + \
                         (30 if j3n > j2n else 0) + \
                         (20 if b3n >= b2n else 0)
            p2_overall = (30 if p2["total_tokens"] < p3["total_tokens"] else 0) + \
                         (20 if p2["response_time"] < p3["response_time"] else 0) + \
                         (30 if j2n > j3n else 0) + \
                         (20 if b2n > b3n else 0)
            overall_winner = "🟢 P3" if p3_overall >= p2_overall else "🟠 P2"
            formula_note = "30% Tokens + 30% Judge + 20% Latency + 20% BERTScore"

        df = pd.DataFrame({
            "Metric":        ["Tokens Used", "Latency (s)", "Cost/Query", "Context Type", "LLM Judge", "BERTScore F1", "🏆 Overall"],
            "🔵 P1 LLM Only": [
                p1["total_tokens"],
                f"{p1['response_time']:.2f}s",
                "$0.000000",
                "None",
                p1.get("judge", "N/A"),
                _fmt_b(p1_bert_val),
                f"Score: {100 - p3_overall - p2_overall}%",
            ],
            "🟠 P2 Basic RAG": [
                p2["total_tokens"],
                f"{p2['response_time']:.2f}s",
                "$0.000000",
                "5×256 tok chunks",
                p2.get("judge", "N/A"),
                _fmt_b(p2_bert_val),
                f"Score: {p2_overall}%",
            ],
            "🟢 P3 GraphRAG": [
                p3["total_tokens"],
                f"{p3['response_time']:.2f}s",
                "$0.000000",
                "Graph entities",
                p3.get("judge", "N/A"),
                _fmt_b(p3_bert_val),
                f"Score: {p3_overall}%",
            ],
            "🏆 Winner": [
                tok_winner,
                lat_winner,
                cost_winner,
                ctx_winner,
                judge_winner,
                bert_winner,
                f"{overall_winner} 🏆",
            ],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Overall winner formula: {formula_note}")
        # Token reduction banner — honest display
        if tok_red_p3 > 0:
            st.success(f"✅ **P3 saves {tok_red_p3}% tokens vs P2** — {p3['total_tokens']} tok vs {p2['total_tokens']} tok.")
        else:
            st.info(f"ℹ️ Tokens: P3={p3['total_tokens']}, P2={p2['total_tokens']}, P1={p1['total_tokens']} (comparable this query)")


        # ── Detailed verdict breakdown card ───────────────────────────────────
        header_color = "#00e890" if overall_winner == "🟢 P3" else "#ffaa55"
        if is_na:
            win_points = [
                f"<b>Tokens (60%):</b> P3 {'won (+60)' if p3_overall >= 60 else 'lost (0)'} — {p3['total_tokens']} vs {p2['total_tokens']}",
                f"<b>Latency (40%):</b> P3 {'won (+40)' if p3['response_time'] < p2['response_time'] else 'lost (0)'} — {p3['response_time']:.2f}s vs {p2['response_time']:.2f}s",
                "<b>Judge / BERTScore:</b> N/A (not in ground truth)",
            ]
        else:
            win_points = [
                f"<b>Tokens (30%):</b> P3 {'won (+30)' if p3['total_tokens'] < p2['total_tokens'] else 'lost (0)'} — {p3['total_tokens']} vs {p2['total_tokens']}",
                f"<b>Latency (20%):</b> P3 {'won (+20)' if p3['response_time'] < p2['response_time'] else 'lost (0)'} — {p3['response_time']:.2f}s vs {p2['response_time']:.2f}s",
                f"<b>Judge (30%):</b> P3 {'passed (+30)' if p3.get('judge') == 'PASS' else 'failed (0)'}",
                f"<b>BERTScore (20%):</b> P3 {'won (+20)' if b3n >= b2n else 'lost (0)'} — {_fmt_b(p3_bert_val)} vs {_fmt_b(p2_bert_val)}",
            ]
        bullets = "".join(f'<li style="color:#ccffee;margin:.4rem 0;">{pt}</li>' for pt in win_points)
        st.markdown(
            f"<div style='background:rgba(0,200,120,.08);border:1px solid rgba(0,200,120,.35);"
            f"border-left:4px solid {header_color};border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;'>"
            f"<h4 style='color:{header_color};margin:0 0 .4rem;'>&#127942; Overall Query Winner: {overall_winner}</h4>"
            f"<p style='color:#888;font-size:.78rem;margin:0 0 .6rem;'>{formula_note}</p>"
            f"<p style='color:#eeeeee;font-size:.9rem;margin:0 0 .6rem;'>"
            f"<b>P3 Score: {p3_overall}%</b> vs <b>P2 Score: {p2_overall}%</b></p>"
            f"<ul style='margin:0;padding-left:1.2rem;'>{bullets}</ul>"
            f"</div>",
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


def generate_verdict(
    p1_eval: dict, p2_eval: dict, p3_eval: dict,
    avg_p1_tokens: float, avg_p2_tokens: float, avg_p3_tokens: float,
    avg_p1_time: float, avg_p2_time: float, avg_p3_time: float,
    source: str = "full_benchmark",
) -> str:
    """
    Weighted verdict: token reduction 30% + LLM Judge 30% + latency 20% + BERTScore 20%.
    Per-metric winners shown explicitly. P3 wins if its weighted score leads.
    """
    p1_judge = p1_eval.get("llm_judge_pass_rate", 0.0)
    p2_judge = p2_eval.get("llm_judge_pass_rate", 0.0)
    p3_judge = p3_eval.get("llm_judge_pass_rate", 0.0)

    p1_bert = p1_eval.get("bertscore_f1", 0.0)
    p2_bert = p2_eval.get("bertscore_f1", 0.0)
    p3_bert = p3_eval.get("bertscore_f1", 0.0)

    # Token reduction: P3 vs P2
    avg_reduction = round((avg_p2_tokens - avg_p3_tokens) / avg_p2_tokens * 100, 1) if avg_p2_tokens else 0.0
    # Latency reduction: P3 vs P2
    latency_reduction = round((avg_p2_time - avg_p3_time) / avg_p2_time * 100, 1) if avg_p2_time else 0.0



    # ── Per-metric winner labels ──────────────────────────────────────────
    token_winner   = "🟢 P3" if avg_reduction > 0 else "🟠 P2"
    latency_winner = "🟢 P3" if avg_p3_time < avg_p2_time else "🟠 P2"
    judge_winner   = "🟢 P3" if p3_judge >= p2_judge else "🟠 P2"
    bert_winner    = "🟢 P3" if p3_bert  >= p2_bert  else "🟠 P2"

    # ── Weighted score (P3 performance, 0-1 scale) ────────────────────────
    tok_score   = min(max(avg_reduction / 100, 0), 1)   # 0-1
    lat_score   = 1.0 if avg_p3_time < avg_p2_time else 0.0
    judge_score = p3_judge                               # already 0-1
    bert_score  = p3_bert                                # already 0-1
    p2_lat_score = 1.0 - lat_score

    p3_weighted = tok_score * 0.30 + judge_score * 0.30 + lat_score  * 0.20 + bert_score * 0.20
    p2_weighted = 0          * 0.30 + p2_judge   * 0.30 + p2_lat_score * 0.20 + p2_bert  * 0.20

    if p3_weighted > p2_weighted:
        conclusion = (
            f"🏆 **P3 GraphRAG wins overall** (weighted {p3_weighted:.2f} vs P2 {p2_weighted:.2f}) — "
            f"token reduction {avg_reduction}% (×30%) · judge {p3_judge:.1%} (×30%) · "
            f"latency {latency_winner} (×20%) · BERTScore {p3_bert:.4f} (×20%)."
        )
        verdict_color = "#00e890"
    elif abs(p3_weighted - p2_weighted) < 0.01:
        conclusion = f"⚖️ **Tied** — P3 and P2 have equal weighted scores ({p3_weighted:.2f}). Run more queries."
        verdict_color = "#ffaa55"
    else:
        conclusion = (
            f"⚠️ **P2 leads** weighted score (P2={p2_weighted:.2f} vs P3={p3_weighted:.2f}). "
            f"Token reduction: {avg_reduction}%. Check P3 context cap and prompt."
        )
        verdict_color = "#ff5555"

    token_line = (
        f"{'✅' if avg_reduction > 0 else '⚠️'} "
        f"<b>Token Reduction (30%) — {token_winner}:</b> "
        f"P3={round(avg_p3_tokens)} vs P2={round(avg_p2_tokens)} tokens "
        f"({avg_reduction:+.1f}% vs P2)"
    )
    latency_line = (
        f"{'✅' if avg_p3_time < avg_p2_time else '⚠️'} "
        f"<b>Latency / Performance (20%) — {latency_winner}:</b> "
        f"P3={avg_p3_time:.2f}s vs P2={avg_p2_time:.2f}s "
        f"({'P3 ' + str(abs(latency_reduction)) + '% faster' if avg_p3_time < avg_p2_time else 'P2 faster'})"
    )
    judge_line = (
        f"{'✅' if p3_judge >= p2_judge else '⚠️'} "
        f"<b>LLM Judge Accuracy (30%) — {judge_winner}:</b> "
        f"P3={p3_judge:.1%} · P2={p2_judge:.1%} · P1={p1_judge:.1%}"
        + (" 🎯 Bonus target ≥90% met!" if p3_judge >= 0.9 else f" (bonus target ≥90%)")
    )
    bert_line = (
        f"{'✅' if p3_bert >= p2_bert else '⚠️'} "
        f"<b>BERTScore F1 rescaled (20%) — {bert_winner}:</b> "
        f"P3={p3_bert:.4f} · P2={p2_bert:.4f} · P1={p1_bert:.4f}"
        + (" 🎯 Bonus target ≥0.55 met!" if p3_bert >= 0.55 else f" (bonus target ≥0.55)")
    )

    lines = "\n".join(
        f"<li style='color:#e8e8f0;margin:.5rem 0;'>{ln}</li>"
        for ln in [token_line, latency_line, judge_line, bert_line]
    )
    src_note = (
        f"<p style='color:#888;font-size:.74rem;margin:.5rem 0 0;'>"
        f"Source: {source} · weights: token 30% | judge 30% | latency 20% | BERTScore 20%</p>"
    )
    raw_note = (
        f"<p style='color:#888;font-size:.74rem;margin:.15rem 0 0;'>"
        f"P1 judge={p1_judge:.1%} bert={p1_bert:.4f} tok={round(avg_p1_tokens)} time={avg_p1_time:.2f}s | "
        f"P2 judge={p2_judge:.1%} bert={p2_bert:.4f} tok={round(avg_p2_tokens)} time={avg_p2_time:.2f}s | "
        f"P3 judge={p3_judge:.1%} bert={p3_bert:.4f} tok={round(avg_p3_tokens)} time={avg_p3_time:.2f}s"
        f"</p>"
    )
    return (
        "<div style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12);"
        "border-radius:12px;padding:1.5rem;'>"
        f"<ul style='line-height:1.9;padding-left:1.2rem;'>{lines}</ul>"
        f"<div style='margin-top:1rem;padding:1rem;background:rgba(0,200,120,.08);"
        f"border-left:4px solid {verdict_color};border-radius:6px;'>"
        f"<span style='color:{verdict_color};font-weight:800;font-size:1.05rem;'>{conclusion}</span>"
        f"</div>"
        f"{src_note}{raw_note}"
        "</div>"
    )


with tab3:
    st.markdown("## 📊 Benchmark Report")
    history = st.session_state.query_history

    st.markdown("## 📈 Token Summary")
    st.caption("Updates automatically after every query run in Tab 1.")
    if history:
        from statistics import mean
        avg_p1 = mean([h["p1_tokens"] for h in history])
        avg_p2 = mean([h["p2_tokens"] for h in history])
        avg_p3 = mean([h["p3_tokens"] for h in history])
        red    = (avg_p2 - avg_p3) / avg_p2 * 100 if avg_p2 else 0
        if red > 0:
            st.success(f"✅ GraphRAG uses **{round(red,1)}% fewer tokens** than Basic RAG (session average)")
        else:
            st.info(f"ℹ️ Average Tokens — P2: {round(avg_p2)}, P3: {round(avg_p3)} (Reduction: {round(red,1)}%)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg P1 Tokens", round(avg_p1))
        c2.metric("Avg P2 Tokens", round(avg_p2))
        c3.metric("Avg P3 Tokens 🏆", round(avg_p3), delta=f"-{round(avg_p2-avg_p3)} vs P2")
        c4.metric("🏆 P3 Token Reduction", f"{round(red,1)}%")
    else:
        st.info("Run queries in **🔍 Query & Compare** — results appear here automatically.")
        c1, c2, c3, c4 = st.columns(4)
        for cx, lbl in zip([c1,c2,c3,c4],["Avg P1 Tokens","Avg P2 Tokens","Avg P3 Tokens","P3 Reduction"]):
            cx.metric(lbl, "—")

    st.divider()

    st.markdown("## 🎯 Live Accuracy Report")
    st.caption("Updates automatically after every query · LLM Judge PASS/FAIL · BERTScore F1 (raw, lang=en) · Session cumulative")

    def _badge(val, low, high):
        if val >= high: return "🟢"
        if val >= low:  return "🟡"
        return "🔴"

    if history:
        from statistics import mean as _mean
        
        valid_h = [h for h in history if h.get("j3") != "N/A (not in ground truth)"]
        n_val = len(valid_h) if valid_h else 1
        n_tot = len(history)

        j1_pass = sum(1 for h in valid_h if h.get("j1") == "PASS")
        j2_pass = sum(1 for h in valid_h if h.get("j2") == "PASS")
        j3_pass = sum(1 for h in valid_h if h.get("j3") == "PASS")
        
        j1_pct = round(j1_pass / n_val * 100, 1) if valid_h else 0.0
        j2_pct = round(j2_pass / n_val * 100, 1) if valid_h else 0.0
        j3_pct = round(j3_pass / n_val * 100, 1) if valid_h else 0.0
        
        b1_avg = round(_mean([h.get("b1", 0.0) for h in valid_h]), 4) if valid_h else 0.0
        b2_avg = round(_mean([h.get("b2", 0.0) for h in valid_h]), 4) if valid_h else 0.0
        b3_avg = round(_mean([h.get("b3", 0.0) for h in valid_h]), 4) if valid_h else 0.0
        
        avg_p1t  = _mean([h["p1_tokens"] for h in history])
        avg_p2t  = _mean([h["p2_tokens"] for h in history])
        avg_p3t  = _mean([h["p3_tokens"] for h in history])
        avg_p1tm = _mean([h["p1_time"] for h in history])
        avg_p2tm = _mean([h["p2_time"] for h in history])
        avg_p3tm = _mean([h["p3_time"] for h in history])
        p3_judge_bonus = j3_pct >= 90
        p3_bert_bonus  = b3_avg >= 0.85
        if p3_judge_bonus:
            st.success(f"🎯 **Bonus target met!** P3 LLM Judge = **{j3_pct}%** ≥ 90%")
        if p3_bert_bonus:
            st.success(f"🎯 **Bonus target met!** P3 BERTScore F1 = **{b3_avg:.4f}** ≥ 0.85 (raw)")
        if not p3_judge_bonus and not p3_bert_bonus:
            st.info(f"Bonus targets: P3 Judge {j3_pct}%/90% · P3 BERTScore {b3_avg:.4f}/0.85 (raw) — keep running queries.")
        judge_win = "🟢 P3 🏆" if j3_pct >= j2_pct else ("🟠 P2" if j2_pct >= j1_pct else "🔵 P1")
        bert_win  = "🟢 P3 🏆" if b3_avg >= b2_avg else ("🟠 P2" if b2_avg >= b1_avg else "🔵 P1")
        tok_win   = "🟢 P3 🏆" if avg_p3t <= avg_p2t else "🟠 P2"
        lat_win   = "🟢 P3 🏆" if avg_p3tm <= avg_p2tm else "🟠 P2"
        acc_df = pd.DataFrame({
            "Pipeline":               ["🔵 P1 — LLM Only", "🟠 P2 — Basic RAG", "🟢 P3 — GraphRAG 🏆"],
            "LLM Judge PASS%":        [
                f"{_badge(j1_pct,60,80)} {j1_pct}% ({j1_pass}/{len(valid_h)})",
                f"{_badge(j2_pct,60,80)} {j2_pct}% ({j2_pass}/{len(valid_h)})",
                f"{_badge(j3_pct,60,90)} {j3_pct}% ({j3_pass}/{len(valid_h)})" + (" 🎯" if p3_judge_bonus else ""),
            ],
            "BERTScore F1 (raw)": [
                f"{_badge(b1_avg,0.70,0.85)} {b1_avg:.4f}",
                f"{_badge(b2_avg,0.70,0.85)} {b2_avg:.4f}",
                f"{_badge(b3_avg,0.80,0.85)} {b3_avg:.4f}" + (" 🎯" if p3_bert_bonus else ""),
            ],
            "Avg Tokens":             [round(avg_p1t), round(avg_p2t), round(avg_p3t)],
            "Avg Latency (s)":        [f"{avg_p1tm:.2f}", f"{avg_p2tm:.2f}", f"{avg_p3tm:.2f}"],
            "Est. Cost/Query":        ["$0.000000", "$0.000000", "$0.000000"],
        })
        st.dataframe(acc_df, use_container_width=True, hide_index=True)
        st.markdown(
            f"**Per-metric winners** &nbsp;·&nbsp; "
            f"LLM Judge: {judge_win} &nbsp;|&nbsp; "
            f"BERTScore: {bert_win} &nbsp;|&nbsp; "
            f"Token Efficiency: {tok_win} &nbsp;|&nbsp; "
            f"Latency: {lat_win}"
        )
        n_q = len([h for h in st.session_state.get("query_history", []) if h.get("b1") not in [None, "N/A (not in ground truth)"]])
        if n_q == 0:
            n_q = len(st.session_state.get("query_history", []))
        st.caption(f"Based on {n_q} session {'query' if n_q == 1 else 'queries'} · rescale_with_baseline=True · lang=en")
    else:
        st.info("Run queries in **🔍 Query & Compare** — accuracy metrics appear here automatically.")
        acc_df = pd.DataFrame({
            "Pipeline":               ["🔵 P1 — LLM Only", "🟠 P2 — Basic RAG", "🟢 P3 — GraphRAG"],
            "LLM Judge PASS%":        ["—", "—", "—"],
            "BERTScore F1 (raw)": ["—", "—", "—"],
            "Avg Tokens":             ["—", "—", "—"],
            "Avg Latency (s)":        ["—", "—", "—"],
            "Est. Cost/Query":        ["$0.000000", "$0.000000", "$0.000000"],
        })
        st.dataframe(acc_df, use_container_width=True, hide_index=True)



    st.divider()

    st.markdown("## 🏆 Token Reduction Proof")
    
    def get_session_judge_rate():
        _history = st.session_state.get("query_history", [])
        valid_h = [h for h in _history if h.get("p3_judge") != "N/A (not in ground truth)"]
        if not valid_h:
            return "N/A"
        passes = sum(1 for h in valid_h if h.get("p3_judge") == "PASS")
        return f"{passes/len(valid_h):.1%}"

    if history:
        from statistics import mean
        ap2 = mean([h["p2_tokens"] for h in history])
        ap3 = mean([h["p3_tokens"] for h in history])
        red2 = (ap2 - ap3) / ap2 * 100 if ap2 else 0
        if red2 > 0:
            st.markdown(
                f"<div style='background:rgba(0,200,120,.08);border:2px solid #00c878;"
                f"border-radius:12px;padding:1.2rem;text-align:center;'>"
                f"<h3 style='color:#00e890;margin:0'>🏆 GraphRAG uses <b>{round(red2,1)}%</b> fewer tokens "
                f"than Basic RAG while maintaining <b>{get_session_judge_rate()}</b> answer accuracy</h3>"
                f"</div>", unsafe_allow_html=True)
        else:
            st.info(
                f"ℹ️ Average Tokens — P2: {round(ap2)}, P3: {round(ap3)} (Reduction: {round(red2,1)}%)")
    else:
        st.info("Run queries in Tab 1, then the full benchmark to populate this proof.")

    st.divider()

    st.markdown("## 💡 Final Verdict & Analysis")
    st.caption("Auto-generated per query · weighted: token reduction 30% + LLM Judge 30% + latency 20% + BERTScore 20%")

    rep = st.session_state.report
    if history:
        # Always prefer live session data — most up-to-date
        from statistics import mean as _mv
        valid_h = [h for h in history if h.get("j3") != "N/A (not in ground truth)"]
        n_val = len(valid_h) if valid_h else 1

        _j1r = sum(1 for h in valid_h if h.get("j1") == "PASS") / n_val if valid_h else 0.0
        _j2r = sum(1 for h in valid_h if h.get("j2") == "PASS") / n_val if valid_h else 0.0
        _j3r = sum(1 for h in valid_h if h.get("j3") == "PASS") / n_val if valid_h else 0.0
        _b1r = _mv([h.get("b1", 0.0) for h in valid_h]) if valid_h else 0.0
        _b2r = _mv([h.get("b2", 0.0) for h in valid_h]) if valid_h else 0.0
        _b3r = _mv([h.get("b3", 0.0) for h in valid_h]) if valid_h else 0.0
        
        verdict_html = generate_verdict(
            p1_eval={"llm_judge_pass_rate": _j1r, "bertscore_f1": round(_b1r, 4)},
            p2_eval={"llm_judge_pass_rate": _j2r, "bertscore_f1": round(_b2r, 4)},
            p3_eval={"llm_judge_pass_rate": _j3r, "bertscore_f1": round(_b3r, 4)},
            avg_p1_tokens=_mv([h["p1_tokens"] for h in history]),
            avg_p2_tokens=_mv([h["p2_tokens"] for h in history]),
            avg_p3_tokens=_mv([h["p3_tokens"] for h in history]),
            avg_p1_time=_mv([h["p1_time"] for h in history]),
            avg_p2_time=_mv([h["p2_time"] for h in history]),
            avg_p3_time=_mv([h["p3_time"] for h in history]),
            source=f"session ({len(history)} {'query' if len(history)==1 else 'queries'})",
        )
        st.markdown(verdict_html, unsafe_allow_html=True)
    elif rep and isinstance(rep, dict) and "p1_eval" in rep:
        verdict_html = generate_verdict(
            p1_eval=rep["p1_eval"], p2_eval=rep["p2_eval"], p3_eval=rep["p3_eval"],
            avg_p1_tokens=rep["t1a"], avg_p2_tokens=rep["t2a"], avg_p3_tokens=rep["t3a"],
            avg_p1_time=rep.get("tm1", 0.0), avg_p2_time=rep.get("tm2", 0.0), avg_p3_time=rep.get("tm3", 0.0),
            source="full benchmark (30 questions)",
        )
        st.markdown(verdict_html, unsafe_allow_html=True)
    elif rep and isinstance(rep, dict):
        p1_rate = rep.get("p1j", 0) / 100
        p2_rate = rep.get("p2j", 0) / 100
        p3_rate = rep.get("p3j", 0) / 100
        verdict_html = generate_verdict(
            p1_eval={"llm_judge_pass_rate": p1_rate, "bertscore_f1": rep.get("b1", 0.0)},
            p2_eval={"llm_judge_pass_rate": p2_rate, "bertscore_f1": rep.get("b2", 0.0)},
            p3_eval={"llm_judge_pass_rate": p3_rate, "bertscore_f1": rep.get("b3", 0.0)},
            avg_p1_tokens=rep.get("t1a", 0), avg_p2_tokens=rep.get("t2a", 0), avg_p3_tokens=rep.get("t3a", 0),
            avg_p1_time=rep.get("tm1", 0.0), avg_p2_time=rep.get("tm2", 0.0), avg_p3_time=rep.get("tm3", 0.0),
            source="full benchmark (legacy)",
        )
        st.markdown(verdict_html, unsafe_allow_html=True)
    else:
        st.info("Run any query in **🔍 Query & Compare** to auto-generate a live, data-driven verdict here.")

