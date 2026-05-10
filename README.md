# 🧠 GraphRAG Inference System

[![TigerGraph](https://img.shields.io/badge/TigerGraph-Savanna_Cloud-orange?style=flat-square&logo=tigergraph)](https://tgcloud.io/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-red?style=flat-square)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blueviolet?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Wikipedia](https://img.shields.io/badge/Dataset-Wikipedia_2.3M_tokens-lightgrey?style=flat-square)](https://pypi.org/project/wikipedia/)

> **A production-ready GraphRAG (Retrieval-Augmented Generation) system built for the TigerGraph Hackathon. Compares three inference pipelines side-by-side — LLM-Only, FAISS Wikipedia RAG, and TigerGraph GraphRAG — powered by a 2.3M-token Wikipedia knowledge base and Groq ultra-fast inference.**

---

## ✅ Hackathon Round 1 Requirements Checklist

| Requirement | Status | Implementation |
|---|---|---|
| TigerGraph on Savanna Cloud | ✅ | `graph/graph_engine.py` — connects via `pyTigerGraph` API |
| FAISS index committed to repo | ✅ | `data/faiss_index.bin` included |
| Pipeline 1: LLM-Only | ✅ | `inference/pipelines.py` → `run_pipeline_1_llm_only()` |
| Pipeline 2: Basic RAG (FAISS) | ✅ | `inference/pipelines.py` → `BasicRAG` class + `run_pipeline_2_basic_rag()` |
| Pipeline 3: GraphRAG (TigerGraph) | ✅ | `inference/pipelines.py` → `run_pipeline_3_graphrag()` + multi-hop traversal |
| Architecture diagram in dashboard | ✅ | Dashboard Tab 2 "Architecture" + `docs/architecture.md` |
| LLM-as-Judge (HF InferenceClient) | ✅ | `eval/benchmark.py` → `llm_judge()` (HF `meta-llama/Llama-3.1-8B-Instruct` + Groq fallback) |
| BERTScore F1 (rescale_with_baseline) | ✅ | `eval/benchmark.py` → `bertscore_eval()` |
| Both metrics shown in dashboard | ✅ | Dashboard Tab 3 "Benchmark Report" |
| Ground truth set (30-50 QA pairs) | ✅ | `eval/benchmark.py` → `GROUND_TRUTH` — 30 hand-written QA pairs |
| Dataset ≥ 2M tokens | ✅ | 230 Wikipedia articles × ~10k tokens avg = **~2.3M raw tokens ingested** |
| Token count documented | ✅ | `eval/benchmark_report.md` Dataset section + this README |
| Token reduction metric (both comparisons) | ✅ | Benchmark report: P2 vs P1 AND P3 vs P2 AND P3 vs P1 |
| One query → all 3 answers + metrics | ✅ | `run_all_three()` — sequential, sub-10s total latency |
| Local fallback if TigerGraph offline | ✅ | `graph/graph_engine.py` → `_local_fallback()` |

---

## ✨ Features

- **3-Pipeline Side-by-Side Comparison**: LLM-Only vs. FAISS Wikipedia RAG vs. TigerGraph GraphRAG
- **2.3M Token Wikipedia Knowledge Base**: 230 Wikipedia articles auto-fetched and chunked with persistent FAISS caching
- **TigerGraph Savanna Cloud**: Upserts entities/relationships and performs 2-hop graph traversal — zero local RAM usage
- **Google Colab Compatible**: Heavy compute (embeddings, BERTScore) can be offloaded to Colab
- **FAISS Index in Repo**: `data/faiss_index.bin` committed for instant startup without rebuild
- **Automated 30-Question Benchmark**: LLM-as-Judge (HF Llama-3.1-8B + Groq fallback) + BERTScore F1 (rescaled)
- **Interactive Streamlit Dashboard**: Dark-themed UI with Query, Architecture, and Benchmark tabs
- **Both Token Reduction Metrics**: P2 vs P1, P3 vs P2, P3 vs P1 all reported

---

## 🏗️ Architecture — 3 Pipelines

```
User Question
      │
      ├──► [Pipeline 1] LLM-Only                       (inference/pipelines.py)
      │         └── Groq API (LLaMA 3.1-8B) ──► Answer
      │
      ├──► [Pipeline 2] Basic RAG  (Wikipedia FAISS)   (inference/pipelines.py)
      │         ├── Encode Question → 384-dim Embedding (MiniLM-L6-v2)
      │         ├── FAISS Similarity Search → Top-3 Wikipedia Chunks
      │         └── Groq API + Retrieved Context ──► Answer
      │
      └──► [Pipeline 3] GraphRAG  (TigerGraph Cloud)   (graph/graph_engine.py)
                ├── Keyword Match → Seed Entity IDs
                ├── TigerGraph Savanna — 2-hop Multi-hop Traversal
                │     └── [offline] Local KB Fallback (_local_fallback)
                └── Groq API + Graph Context ──► Answer
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Wikipedia (via `wikipedia` Python package) |
| Articles fetched | ~230 (full topic list: AI, ML, NLP, Graph DBs, Science, History) |
| Raw tokens ingested | **~2.3M** (230 articles × ~10,000 tokens avg) |
| FAISS chunks | ~4,600 chunks × 256 tokens each |
| Entity/Relationship pairs | ~225 entities, ~275 relationships |
| Token type | **Raw text ingested** (not LLM inference tokens) |

---

## 📁 Project Structure

```text
graphrag_project/
│
├── config.py                    # Central config: GROQ_API_KEY, TigerGraph creds from .env
├── main.py                      # Entry point: validates config, runs all 3 pipelines on test query
├── requirements.txt             # All Python dependencies
├── .env                         # Your credentials (gitignored)
├── .env.example                 # Template — copy to .env
│
├── data/
│   ├── __init__.py              # Package marker (Python requirement — no logic)
│   ├── knowledge.py             # Wikipedia loader + entity extractor + fallback KB
│   ├── faiss_index.bin          # Pre-built FAISS vector index (committed to repo)
│   ├── faiss_chunks.pkl         # Chunk list aligned to FAISS index
│   ├── wiki_chunks.pkl          # Wikipedia chunk cache
│   └── wiki_entities.pkl        # Wikipedia entity/relationship cache
│
├── graph/
│   ├── __init__.py              # Package marker (Python requirement — no logic)
│   └── graph_engine.py          # TigerGraph: connect → schema → load → multi-hop traversal
│                                  # + _local_fallback() for offline mode
│                                  # + _KEYWORD_MAP for entity routing
│
├── inference/
│   ├── __init__.py              # Package marker (Python requirement — no logic)
│   └── pipelines.py             # All 3 inference pipelines:
│                                  # · run_pipeline_1_llm_only()
│                                  # · run_pipeline_2_basic_rag()  [BasicRAG class]
│                                  # · run_pipeline_3_graphrag()
│                                  # · run_all_three()  ← main orchestrator
│
├── eval/
│   ├── __init__.py              # Package marker (Python requirement — no logic)
│   ├── benchmark.py             # 30-question evaluation suite:
│                                  # · GROUND_TRUTH[30]  (factual + multi-hop + synthesis)
│                                  # · llm_judge()  (HF meta-llama + Groq fallback)
│                                  # · bertscore_eval()  (rescale_with_baseline=True)
│                                  # · evaluate_all_pipelines()
│                                  # · run_benchmark()
│   └── benchmark_report.md      # Auto-generated: tokens, latency, judge%, BERTScore, token-reduction
│
├── dashboard/
│   ├── __init__.py              # Package marker (Python requirement — no logic)
│   └── app.py                   # Streamlit dashboard (3 tabs):
│                                  # Tab 1: Query & Compare — live 3-pipeline query
│                                  # Tab 2: Architecture — system diagram
│                                  # Tab 3: Benchmark — 30Q eval + report generation
│
└── docs/
    └── architecture.md          # Detailed system architecture documentation
```

> **Why `__init__.py`?** Python requires these files to exist in every directory imported as a package (e.g., `from data.knowledge import ...` needs `data/__init__.py`). They contain no logic — only a 2-line comment. They **must** be named `__init__.py`; this is a Python language rule, not a design choice.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_key           # Free: console.groq.com/keys
TIGERGRAPH_HOST=your_savanna_host    # Optional: tgcloud.io (uses local fallback if missing)
TIGERGRAPH_PASSWORD=your_password
TIGERGRAPH_GRAPH_NAME=GraphRAGDemo
HF_TOKEN=your_hf_token              # Optional: for HF LLM-as-Judge
```

### 3. Initialize & Test (runs all 3 pipelines)

```bash
python main.py
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501**

### 5. Run Full 30-Question Benchmark

```bash
python eval/benchmark.py
```

Results saved to `eval/benchmark_report.md`.

---

## 📦 Key Dependencies

| Library | Role |
|---|---|
| `groq` | LLaMA 3.1-8B inference via Groq API |
| `faiss-cpu` | FAISS vector index (committed to repo) |
| `sentence-transformers` | MiniLM-L6-v2 text → embedding |
| `wikipedia` | Live 2.3M-token Wikipedia knowledge base |
| `pyTigerGraph` | TigerGraph Savanna Cloud connection |
| `streamlit` | Interactive dashboard |
| `evaluate` + `bert-score` | BERTScore F1 (rescale_with_baseline=True) |
| `huggingface_hub` | HF InferenceClient for LLM-as-Judge |

---

## 🔁 Fallback Architecture

If TigerGraph Savanna is unreachable:
1. `graph/graph_engine.py` catches the DNS/connection error and logs it
2. `find_relevant_context()` routes to `_local_fallback()`
3. Fallback uses **25 hardcoded AI/ML/Graph entities** embedded in `data/knowledge.py`
4. Same multi-hop traversal algorithm, fully in-memory — no network needed
5. Dashboard shows **"Local Wikipedia KB"** tag to distinguish modes

---

## 📄 License

MIT License.

*Built with ❤️ for the TigerGraph Hackathon.*
