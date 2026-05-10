# docs/architecture.md - System Architecture

# GraphRAG Inference System Architecture

## Overview

This system implements a **3-pipeline inference benchmark** comparing LLM-Only, Basic RAG, and GraphRAG approaches for answering questions over a structured knowledge base.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY INPUT                          │
│                    "What is machine learning?"                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE ORCHESTRATOR                         │
│                 inference/orchestrator.py                         │
│                    run_all_three(question)                        │
└────────────┬───────────────────┬───────────────────┬────────────┘
             │                   │                   │
             ▼                   ▼                   ▼
     ┌───────────────┐  ┌────────────────┐  ┌───────────────────┐
     │  PIPELINE 1   │  │  PIPELINE 2    │  │    PIPELINE 3     │
     │   LLM Only    │  │  Basic RAG     │  │     GraphRAG      │
     │ llm/caller.py │  │ rag/basic_rag  │  │  graph/query.py   │
     └───────┬───────┘  └───────┬────────┘  └────────┬──────────┘
             │                  │                     │
             │           ┌──────┴──────┐      ┌──────┴───────┐
             │           │    FAISS    │      │  TigerGraph  │
             │           │   Vector   │      │  Multi-hop   │
             │           │   Index    │      │  Traversal   │
             │           │ (35 chunks)│      │   (GSQL)     │
             │           └──────┬──────┘      └──────┬───────┘
             │                  │                     │
             │           ┌──────┴──────┐      ┌──────┴───────┐
             │           │  sentence-  │      │   Local KB   │
             │           │ transformers│      │  (Fallback)  │
             │           │all-MiniLM  │      │ knowledge.py │
             │           └─────────────┘      └──────────────┘
             │                  │                     │
             └──────────────────┴─────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │       GROQ LLaMA3 API        │
                  │    llama-3.1-8b-instant      │
                  │   api.groq.com (free tier)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │      EVALUATION LAYER        │
                  │   eval/accuracy.py           │
                  │                              │
                  │  ┌──────────────────────┐   │
                  │  │  LLM-as-Judge        │   │
                  │  │  HuggingFace API     │   │
                  │  │  (Groq fallback)     │   │
                  │  └──────────────────────┘   │
                  │  ┌──────────────────────┐   │
                  │  │  BERTScore (F1)      │   │
                  │  │  evaluate library    │   │
                  │  └──────────────────────┘   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │     STREAMLIT DASHBOARD      │
                  │    dashboard/app.py           │
                  │                              │
                  │  • 3-column answer display   │
                  │  • Performance metrics table  │
                  │  • Accuracy evaluation panel  │
                  │  • Context expanders          │
                  │  • Query history              │
                  └─────────────────────────────┘
```

---

## Component Details

### Pipeline 1 — LLM Only
- **File:** `llm/caller.py` → `call_llm_baseline()`
- **Flow:** Question → Groq API → Answer
- **Purpose:** Baseline — no retrieval augmentation
- **Strength:** Fast, zero-latency retrieval
- **Weakness:** Relies entirely on parametric memory, prone to hallucination

### Pipeline 2 — Basic RAG
- **File:** `rag/basic_rag.py` → `BasicRAG.answer()`
- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS flat L2 index (35 pre-embedded chunks)
- **Flow:** Question → Embed → FAISS top-3 search → Context → Groq → Answer
- **Strength:** Semantic retrieval over curated knowledge chunks
- **Weakness:** Flat retrieval misses relational multi-hop connections

### Pipeline 3 — GraphRAG
- **File:** `graph/query.py` → `find_relevant_context()`
- **Graph DB:** TigerGraph Cloud (local fallback via `data/knowledge.py`)
- **Flow:** Question → Entity extraction → TigerGraph multi-hop → Context → Groq → Answer
- **Strength:** Traverses relationships, captures multi-hop reasoning
- **Weakness:** Higher latency, requires TigerGraph connection

---

## Data Layer

| File | Purpose |
|------|---------|
| `data/knowledge.py` | 25 entities + 26 relationships + 35 text chunks |
| `data/ground_truth.py` | 30 curated Q&A pairs for evaluation |

---

## Evaluation Layer

| Method | Tool | Purpose |
|--------|------|---------|
| LLM-as-Judge | HuggingFace / Groq | PASS/FAIL grading per answer |
| BERTScore F1 | `evaluate` library | Semantic similarity to ground truth |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Graph Database | TigerGraph Cloud (GSQL) |
| Vector Store | FAISS (faiss-cpu) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM Inference | Groq API (llama-3.1-8b-instant) |
| Judge LLM | HuggingFace (Llama-3.1-8B-Instruct) |
| BERTScore | `evaluate` + `bert-score` |
| Dashboard | Streamlit |
| Language | Python 3.10+ |
