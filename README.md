# 🧠 GraphRAG Inference System

[![TigerGraph](https://img.shields.io/badge/TigerGraph-Cloud-orange?style=flat-square&logo=tigergraph)](https://tgcloud.io/)
[![Groq](https://img.shields.io/badge/Groq-API-red?style=flat-square)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://www.python.org/)

> **A comprehensive GraphRAG (Retrieval-Augmented Generation) system built for the TigerGraph Hackathon. It compares a Baseline LLM side-by-side with a Graph-Augmented RAG pipeline using TigerGraph and Groq's high-speed inference.**

---

## ✨ Features

- **Side-by-Side Comparison**: Visually compare answers from a standard Baseline LLM against a Graph-Augmented LLM.
- **TigerGraph Integration**: Stores and queries a rich knowledge base using multi-hop traversals to fetch highly relevant context.
- **Groq LLaMA 3 Inference**: Blazing fast language generation using the free Groq API.
- **Detailed Metrics**: Real-time breakdown of tokens used, response times, cost, and context quality for both pipelines.
- **Local Fallback Mode**: If your TigerGraph cloud instance is asleep, the system gracefully falls back to a local knowledge base.
- **Sleek Streamlit Dashboard**: A professional, dark-themed UI for easy interaction and presentation.

---

## 🏗️ Architecture

1. **Pipeline 1 (Baseline)**: User Question ──► Groq API ──► Answer
2. **Pipeline 2 (GraphRAG)**: User Question ──► Keyword Match ──► TigerGraph Multi-hop Traversal ──► Graph Context ──► Groq API ──► Answer

GraphRAG enriches the LLM with structured knowledge, significantly reducing hallucinations and grounding answers in factual data.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/graphrag_project.git
cd graphrag_project
```

### 2. Set Up Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and configure:
- `GROQ_API_KEY`: Get a free key from [Groq Console](https://console.groq.com/keys)
- `TIGERGRAPH_HOST`: Your TigerGraph Cloud cluster URL
- `TIGERGRAPH_PASSWORD` / `SECRET`: Your TigerGraph instance password

### 3. Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Initialize the Graph

Run the initialization script. This will connect to TigerGraph, create the schema, and load the tech knowledge base.

```bash
python main.py
```
*(Note: If TigerGraph is not connected, it will safely skip to using the local fallback).*

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501** to use the interactive dashboard!

---

## 📁 Project Structure

```text
graphrag_project/
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules
├── config.py                   # Centralized configuration
├── main.py                     # Entry point: DB setup and validation
├── requirements.txt            # Python dependencies
│
├── data/
│   └── knowledge.py            # Local knowledge base (Entities & Relationships)
│
├── graph/
│   ├── connection.py           # TigerGraph connection logic
│   ├── schema.py               # Schema definitions
│   ├── loader.py               # Data injection
│   └── query.py                # Graph traversal algorithms
│
├── inference/
│   └── orchestrator.py         # Pipeline coordination
│
├── llm/
│   └── caller.py               # Groq API wrappers
│
├── eval/
│   └── metrics.py              # Performance tracking
│
└── dashboard/
    └── app.py                  # Streamlit UI
```

---

## 📚 Built-in Knowledge Base
The repository includes a curated tech dataset out-of-the-box featuring **25 entities** and **30+ relationships** spanning:
- Machine Learning & Deep Learning
- Natural Language Processing (NLP) & Transformers
- Graph Technology & Databases
- Frameworks (PyTorch, TensorFlow, Streamlit, etc.)

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request if you'd like to improve the queries, expand the knowledge base, or refine the UI.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

*Built with ❤️ for the TigerGraph Hackathon.*
