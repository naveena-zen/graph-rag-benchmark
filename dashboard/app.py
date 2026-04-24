"""
dashboard/app.py - Streamlit Dashboard
=======================================
The main web interface for the GraphRAG Inference System.
Run with: streamlit run dashboard/app.py

Features:
  - Text input for user question
  - Run Both Pipelines button
  - Side-by-side answer comparison
  - Metrics comparison table
  - Graph context viewer
  - Query history (last 5)
  - Professional dark-themed UI
"""

import sys
import os
import time

# ── Fix Python path so imports work from dashboard/ directory ─────────────────
# This ensures we can import from the project root (config, inference, etc.)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd

# ── Page configuration (must be the very first Streamlit call) ────────────────
st.set_page_config(
    page_title="GraphRAG Inference Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for professional look ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global styles ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Header banner ── */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102,126,234,0.4);
    }
    .main-header h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* ── Pipeline answer cards ── */
    .answer-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    .answer-card:hover {
        transform: translateY(-2px);
    }
    .pipeline1-card {
        border-top: 4px solid #4facfe;
    }
    .pipeline2-card {
        border-top: 4px solid #43e97b;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-body {
        color: rgba(255,255,255,0.88);
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* ── Metrics table ── */
    .metrics-container {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* ── Context card ── */
    .context-card {
        background: rgba(67,233,123,0.06);
        border: 1px solid rgba(67,233,123,0.25);
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.8);
        max-height: 350px;
        overflow-y: auto;
        white-space: pre-wrap;
    }

    /* ── History item ── */
    .history-item {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.8);
    }

    /* ── Badge pills ── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-blue  { background: rgba(79,172,254,0.2); color: #4facfe; border: 1px solid #4facfe40; }
    .badge-green { background: rgba(67,233,123,0.2); color: #43e97b; border: 1px solid #43e97b40; }
    .badge-gold  { background: rgba(252,196,25,0.2); color: #fcc419; border: 1px solid #fcc41940; }

    /* ── Status indicator ── */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.4rem;
    }
    .dot-green  { background: #43e97b; box-shadow: 0 0 6px #43e97b; }
    .dot-yellow { background: #fcc419; box-shadow: 0 0 6px #fcc419; }
    .dot-red    { background: #f64f59; box-shadow: 0 0 6px #f64f59; }

    /* ── Section headings ── */
    .section-heading {
        color: white;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.82rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 3rem;
    }

    /* ── Streamlit widget overrides ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
    }
    .stDataFrame {
        background: transparent !important;
    }
    div[data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.3rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.6) !important;
    }
    .stSidebar {
        background: rgba(15,12,41,0.95) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []   # List of result dicts
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "running" not in st.session_state:
    st.session_state.running = False


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🧠 GraphRAG Inference System</h1>
    <p>Compare Baseline LLM vs Graph-Augmented RAG — Powered by TigerGraph &amp; Groq AI</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ System Status")
    st.divider()

    # Check Groq key
    try:
        import config as cfg
        groq_ok = bool(cfg.GROQ_API_KEY and cfg.GROQ_API_KEY != "your_groq_api_key_here")
        tg_ok     = bool(
            cfg.TIGERGRAPH_HOST     and cfg.TIGERGRAPH_HOST     != "your_tigergraph_host_here" and
            cfg.TIGERGRAPH_PASSWORD and cfg.TIGERGRAPH_PASSWORD != "your_tigergraph_password_here"
        )
    except Exception:
        groq_ok = False
        tg_ok     = False

    groq_status = (
        '<span class="status-dot dot-green"></span>Connected'
        if groq_ok else
        '<span class="status-dot dot-red"></span>API Key Missing'
    )
    tg_status = (
        '<span class="status-dot dot-green"></span>Configured'
        if tg_ok else
        '<span class="status-dot dot-yellow"></span>Using Local Fallback'
    )

    st.markdown(f"**Groq API:** {groq_status}", unsafe_allow_html=True)
    st.markdown(f"**TigerGraph:** {tg_status}", unsafe_allow_html=True)

    if not groq_ok:
        st.warning("⚠️ Add your GROQ_API_KEY to the .env file to enable responses.")

    if not tg_ok:
        st.info("ℹ️ TigerGraph credentials not set. GraphRAG will use local knowledge base for context retrieval.")

    st.divider()
    st.markdown("## 📚 About")
    st.markdown("""
    This system demonstrates **GraphRAG** by:
    1. **Pipeline 1**: Sends your question directly to Groq LLaMA3
    2. **Pipeline 2**: First retrieves context from TigerGraph, then sends question + context to Groq LLaMA3
    
    The comparison shows how graph-structured knowledge improves LLM answers.
    """)

    st.divider()
    st.markdown("## 🔗 Sample Questions")
    sample_questions = [
        "What is machine learning?",
        "How do transformers work?",
        "What is GraphRAG?",
        "Explain deep learning vs machine learning",
        "What is TigerGraph used for?",
        "How does RAG reduce hallucinations?",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state["prefill_question"] = q
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Question Input ────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")

st.markdown('<p class="section-heading">💬 Ask a Question</p>', unsafe_allow_html=True)

question = st.text_area(
    label="Your question:",
    value=prefill if prefill else "What is machine learning?",
    height=100,
    placeholder="e.g. What is machine learning? How do transformers work?",
    key="question_input",
    label_visibility="collapsed",
)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
with col_btn1:
    run_button = st.button("🚀 Run Both Pipelines", key="run_btn", use_container_width=True)
with col_btn2:
    clear_button = st.button("🗑️ Clear History", key="clear_btn", use_container_width=True)

if clear_button:
    st.session_state.query_history  = []
    st.session_state.current_result = None
    st.rerun()


# ── Run Pipelines ─────────────────────────────────────────────────────────────
if run_button:
    if not question.strip():
        st.error("❌ Please enter a question first.")
    elif not groq_ok:
        st.error("❌ GROQ_API_KEY is not set. Please add it to your .env file.")
    else:
        with st.spinner("⚙️ Running both pipelines... Please wait."):
            try:
                from inference.orchestrator import run_both_pipelines
                result = run_both_pipelines(question.strip())

                # Store result
                st.session_state.current_result = result

                # Add to history (keep last 5)
                st.session_state.query_history.insert(0, {
                    "question":  question.strip(),
                    "timestamp": result["timestamp"],
                    "result":    result,
                })
                st.session_state.query_history = st.session_state.query_history[:5]

                st.success("✅ Both pipelines completed!")

            except Exception as e:
                st.error(f"❌ Pipeline error: {str(e)}")
                st.exception(e)


# ── Display Results ───────────────────────────────────────────────────────────
result = st.session_state.current_result

if result:
    p1 = result["pipeline1"]
    p2 = result["pipeline2"]
    m  = result["metrics"]
    gi = result["graph_info"]

    st.divider()

    # ── Quick Metric Cards ────────────────────────────────────────────────────
    st.markdown('<p class="section-heading">📊 Quick Metrics</p>', unsafe_allow_html=True)

    qm1, qm2, qm3, qm4, qm5 = st.columns(5)
    with qm1:
        st.metric("🔵 Baseline Tokens",  m["baseline_total_tokens"])
    with qm2:
        st.metric("🟢 GraphRAG Tokens",  m["graphrag_total_tokens"],
                  delta=f"+{m['token_difference']}" if m['token_difference'] >= 0 else str(m['token_difference']))
    with qm3:
        st.metric("⏱️ Baseline Time",  f"{m['baseline_response_time']}s")
    with qm4:
        st.metric("⏱️ GraphRAG Time",  f"{m['graphrag_response_time']}s")
    with qm5:
        st.metric("🎯 Context Quality", m["context_quality_label"])

    st.divider()

    # ── Side-by-side Answers ──────────────────────────────────────────────────
    st.markdown('<p class="section-heading">🤖 Pipeline Answers</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="answer-card pipeline1-card">
            <div class="card-title">
                🔵 Pipeline 1 — Baseline LLM
                <span class="badge badge-blue">No Context</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-body">{p1["answer"] if not p1["error"] else "❌ Error: " + str(p1["error"])}</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.caption(f"🔑 Tokens: {p1['total_tokens']} | ⏱️ {p1['response_time']}s | 💰 ${p1['cost_usd']:.8f}")

    with col2:
        fallback_note = gi.get("fallback_note", "")
        context_badge = (
            '<span class="badge badge-gold">⚠️ Local Fallback</span>'
            if gi.get("fallback") else
            '<span class="badge badge-green">TigerGraph Context</span>'
        )
        st.markdown(f"""
        <div class="answer-card pipeline2-card">
            <div class="card-title">
                🟢 Pipeline 2 — GraphRAG
                {context_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if fallback_note:
            st.info(fallback_note)
        st.markdown(
            f'<div class="card-body">{p2["answer"] if not p2["error"] else "❌ Error: " + str(p2["error"])}</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.caption(f"🔑 Tokens: {p2['total_tokens']} | ⏱️ {p2['response_time']}s | 💰 ${p2['cost_usd']:.8f} | 🕸️ Nodes: {gi['nodes_found']}")

    st.divider()

    # ── Detailed Metrics Table ────────────────────────────────────────────────
    st.markdown('<p class="section-heading">📈 Detailed Metrics Comparison</p>', unsafe_allow_html=True)

    metrics_data = {
        "Metric":        [
            "Input Tokens",
            "Output Tokens",
            "Total Tokens",
            "Response Time (s)",
            "Cost (USD)",
            "Answer Length (words)",
            "Context Nodes Retrieved",
            "Context Quality",
            "Seed Entities Found",
            "Model Used",
        ],
        "Pipeline 1 — Baseline": [
            str(m["baseline_input_tokens"]),
            str(m["baseline_output_tokens"]),
            str(m["baseline_total_tokens"]),
            f"{m['baseline_response_time']}s",
            f"${m['baseline_cost_usd']:.8f}",
            str(m["baseline_word_count"]),
            "N/A (no graph)",
            "N/A",
            "N/A",
            m["model_used"],
        ],
        "Pipeline 2 — GraphRAG": [
            str(m["graphrag_input_tokens"]),
            str(m["graphrag_output_tokens"]),
            str(m["graphrag_total_tokens"]),
            f"{m['graphrag_response_time']}s",
            f"${m['graphrag_cost_usd']:.8f}",
            str(m["graphrag_word_count"]),
            str(m["nodes_retrieved"]),
            m["context_quality_label"],
            str(m["seed_entities_found"]),
            m["model_used"],
        ],
        "Difference / Note": [
            f"{m['token_difference']:+d}",
            "—",
            f"{m['token_difference']:+d}",
            m["time_note"],
            f"${m['cost_difference_usd']:+.8f}",
            f"{m['graphrag_word_count'] - m['baseline_word_count']:+d} words",
            f"{m['nodes_retrieved']} nodes from graph",
            m["context_quality_label"],
            f"{m['seed_entities_found']} matched",
            "Same model",
        ],
    }

    df = pd.DataFrame(metrics_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Graph Context Viewer ──────────────────────────────────────────────────
    st.markdown('<p class="section-heading">🕸️ Graph Context Used by Pipeline 2</p>', unsafe_allow_html=True)

    if gi.get("context_text"):
        st.markdown(
            f'<div class="context-card">{gi["context_text"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("No context was retrieved from the graph for this query.")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Seed Entities", len(gi.get("seed_entities", [])))
    with col_info2:
        st.metric("Nodes Retrieved", gi.get("nodes_found", 0))
    with col_info3:
        st.metric("Source", "TigerGraph" if not gi.get("fallback") else "Local Fallback")


# ── Query History ─────────────────────────────────────────────────────────────
if st.session_state.query_history:
    st.divider()
    st.markdown('<p class="section-heading">🕐 Query History (Last 5)</p>', unsafe_allow_html=True)

    for i, item in enumerate(st.session_state.query_history):
        h_result = item["result"]
        h_m      = h_result["metrics"]
        col_h1, col_h2 = st.columns([3, 1])

        with col_h1:
            st.markdown(
                f'<div class="history-item">'
                f'<strong>Q{i+1}:</strong> {item["question"]}<br>'
                f'<small>🕐 {item["timestamp"]} | '
                f'Baseline: {h_m["baseline_total_tokens"]} tokens | '
                f'GraphRAG: {h_m["graphrag_total_tokens"]} tokens | '
                f'Quality: {h_m["context_quality_label"]}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_h2:
            if st.button("📋 Load", key=f"load_history_{i}", use_container_width=True):
                st.session_state.current_result = h_result
                st.rerun()


# ── No result yet ─────────────────────────────────────────────────────────────
if not result:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        color: rgba(255,255,255,0.4);
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
        <h3 style="color: rgba(255,255,255,0.6);">Enter a question above and click <em>Run Both Pipelines</em></h3>
        <p>The system will compare a direct LLM response with a graph-augmented response side by side.</p>
        <p style="font-size: 0.85rem; margin-top: 2rem;">
            💡 Try: <em>"What is machine learning?"</em> or <em>"How do transformers work?"</em>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🏆 Built for the <strong>TigerGraph GraphRAG Inference Hackathon</strong> &nbsp;|&nbsp;
    Powered by <strong>TigerGraph</strong> + <strong>Groq AI</strong> &nbsp;|&nbsp;
    Built with <strong>Streamlit</strong>
</div>
""", unsafe_allow_html=True)
