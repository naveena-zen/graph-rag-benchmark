"""
data/knowledge.py - Wikipedia Knowledge Base + Fallback Dataset
================================================================
Primary: Uses `wikipedia` Python package to fetch 600 articles via API.
Chunks each article into 256-token chunks for FAISS/BasicRAG (Pipeline 2).
Extracts entities/relationships from chunks for TigerGraph (Pipeline 3).
Cache saved to data/wiki_chunks.pkl and data/wiki_entities.pkl.

Falls back to hardcoded tech KB if all Wikipedia methods fail.
"""

import os
import sys
import re
import pickle

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── File paths ─────────────────────────────────────────────────────────────────
_DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
_CHUNKS_PKL = os.path.join(_DATA_DIR, "wiki_chunks.pkl")
_ENTS_PKL   = os.path.join(_DATA_DIR, "wiki_entities.pkl")

# ── 200 Wikipedia search topics (AI/ML/Science/Tech/History/etc.) ──────────────
_WIKI_TOPICS = [
    "Machine learning", "Deep learning", "Artificial intelligence",
    "Neural network", "Convolutional neural network", "Recurrent neural network",
    "Transformer (machine learning model)", "BERT (language model)",
    "GPT (language model)", "Natural language processing",
    "Supervised learning", "Unsupervised learning", "Reinforcement learning",
    "Support vector machine", "Random forest", "Gradient boosting",
    "K-means clustering", "Principal component analysis",
    "Backpropagation", "Stochastic gradient descent",
    "Overfitting", "Regularization (mathematics)",
    "Attention mechanism (machine learning)", "Word2vec", "GloVe (machine learning)",
    "Sentiment analysis", "Named entity recognition", "Machine translation",
    "Question answering", "Text summarization",
    "Knowledge graph", "Graph database", "Neo4j", "TigerGraph",
    "Retrieval-augmented generation", "Vector database", "FAISS",
    "Embedding (machine learning)", "Semantic search",
    "Python (programming language)", "TensorFlow", "PyTorch", "Keras",
    "Scikit-learn", "NumPy", "Pandas (software)", "Jupyter Notebook",
    "Big data", "Apache Spark", "Hadoop", "MapReduce",
    "Cloud computing", "Amazon Web Services", "Microsoft Azure", "Google Cloud",
    "Docker (software)", "Kubernetes", "DevOps",
    "Blockchain", "Cryptocurrency", "Bitcoin", "Ethereum",
    "Internet of Things", "Edge computing", "5G",
    "Quantum computing", "Quantum mechanics",
    "Computer vision", "Object detection", "Image segmentation",
    "Face recognition", "Optical character recognition",
    "Speech recognition", "Text-to-speech", "Voice assistant",
    "Autonomous vehicle", "Robotics", "Drone (unmanned aerial vehicle)",
    "Algorithm", "Data structure", "Sorting algorithm", "Binary search",
    "Graph theory", "Dynamic programming", "Big O notation",
    "Database", "SQL", "NoSQL", "PostgreSQL", "MongoDB",
    "Computer network", "Internet protocol", "HTTP", "TCP/IP",
    "Cybersecurity", "Encryption", "Public-key cryptography", "Firewall",
    "Operating system", "Linux", "Windows", "Unix",
    "Compiler", "Interpreter", "Virtual machine",
    "Computer hardware", "Central processing unit", "Graphics processing unit",
    "Memory management", "Cache memory", "Solid-state drive",
    "History of computing", "Alan Turing", "John von Neumann",
    "Ada Lovelace", "Grace Hopper", "Claude Shannon",
    "Calculus", "Linear algebra", "Statistics", "Probability theory",
    "Information theory", "Game theory", "Optimization",
    "Biology", "Genetics", "CRISPR", "Protein folding",
    "Climate change", "Renewable energy", "Solar panel", "Wind turbine",
    "Physics", "Relativity", "Quantum field theory",
    "Chemistry", "Periodic table", "Chemical reaction",
    "Evolution", "Natural selection", "Charles Darwin",
    "Philosophy", "Logic", "Ethics", "Consciousness",
    "Economics", "Supply and demand", "Inflation", "Gross domestic product",
    "Psychology", "Cognitive science", "Neuroscience",
    "Universe", "Galaxy", "Black hole", "Big Bang",
    "Space exploration", "NASA", "SpaceX", "Mars",
    "Democracy", "Capitalism", "Socialism",
    "World War II", "Cold War", "United Nations",
    "Medicine", "Vaccine", "Antibiotics", "Human Genome Project",
    "Language", "Linguistics", "Communication",
    "Art", "Music", "Cinema", "Literature",
    "Mathematics", "Number theory", "Topology", "Abstract algebra",
    "Data science", "Business intelligence", "Data visualization",
    "Social network", "Social media", "Facebook", "Twitter",
    "Search engine", "Google Search", "PageRank",
    "E-commerce", "Supply chain", "Logistics",
    "Recommendation system", "Collaborative filtering",
    "Fraud detection", "Anomaly detection",
    "Computer graphics", "Virtual reality", "Augmented reality",
    "Game development", "Video game", "Simulation",
    "Software engineering", "Agile software development", "Version control",
    "Open source", "Free software", "GitHub",
    "Microservices", "API", "REST", "GraphQL",
    "Distributed computing", "Parallel computing", "High-performance computing",
    "Artificial general intelligence", "Singularity (technological)",
    "Ethics of artificial intelligence", "Bias in algorithms",
    "Privacy", "GDPR", "Data protection",
    "Fintech", "Insurtech", "Healthtech",
    "Startup", "Venture capital", "Silicon Valley",
]


# ── 256-token chunker ──────────────────────────────────────────────────────────

def _chunk_text(text: str, max_tokens: int = 256) -> list:
    """Split text into ~256-token chunks (approx 1 token ≈ 4 chars)."""
    max_chars = max_tokens * 4
    words     = text.split()
    chunks    = []
    current   = []
    current_len = 0
    for word in words:
        wl = len(word) + 1
        if current_len + wl > max_chars and current:
            chunks.append(" ".join(current))
            current     = [word]
            current_len = wl
        else:
            current.append(word)
            current_len += wl
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.strip()) > 60]


# ── Entity extractor ───────────────────────────────────────────────────────────

def _extract_entities_from_chunks(chunks: list, max_entities: int = 200) -> tuple:
    """Lightweight NER: extract capitalized phrases as entities."""
    cap_re   = re.compile(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b')
    skip     = {"This","That","There","These","Those","They","Their","Them",
                "Then","When","Where","Which","While","Also","From","With",
                "Have","Been","Were","Will","More","Some","Such","Each","Many",
                "After","Before","Since","During","Between","However","Although"}
    seen     = {}
    entities = []

    for chunk in chunks[:400]:
        for m in cap_re.finditer(chunk):
            name = m.group(1).strip()
            if len(name) < 4 or name in seen or name in skip:
                continue
            eid = f"wiki_{len(entities):04d}"
            seen[name] = eid
            sentence_re = re.compile(r'[^.!?]*' + re.escape(name) + r'[^.!?]*[.!?]')
            match = sentence_re.search(chunk)
            desc = match.group(0).strip() if match else f"Wikipedia entity: {name}"
            entities.append({
                "id": eid, "name": name, "type": "WikiEntity",
                "description": desc[:400], "domain": "Wikipedia",
            })
            if len(entities) >= max_entities:
                break
        if len(entities) >= max_entities:
            break

    relationships = []
    ent_names = list(seen.keys())
    for chunk in chunks[:150]:
        present = [n for n in ent_names if n in chunk]
        for i in range(len(present) - 1):
            src = seen.get(present[i])
            tgt = seen.get(present[i + 1])
            if src and tgt and src != tgt and len(relationships) < 500:
                relationships.append({
                    "source": src, "target": tgt,
                    "relation": "RELATED_TO",
                    "description": f"{present[i]} co-occurs with {present[i+1]} in Wikipedia",
                })
    return entities, relationships


# ── Wikipedia API loader ───────────────────────────────────────────────────────

def _load_via_wikipedia_package(n_articles: int = 600) -> list:
    """Fetch articles using the `wikipedia` Python package."""
    import wikipedia as wp
    wp.set_lang("en")

    all_chunks = []
    fetched    = 0
    topics     = _WIKI_TOPICS[:n_articles]

    print(f"[Knowledge] Fetching {len(topics)} Wikipedia articles via API ...")
    for i, topic in enumerate(topics):
        try:
            page = wp.page(topic, auto_suggest=True)
            text = page.content or ""
            if len(text) < 100:
                continue
            chunks = _chunk_text(text, max_tokens=256)
            all_chunks.extend(chunks[:20])  # up to 20 chunks per article (~5k tokens each)
            fetched += 1
            if (i + 1) % 50 == 0:
                print(f"   [{i+1}/{len(topics)}] Fetched {fetched} articles, {len(all_chunks)} chunks ...")
        except Exception:
            pass   # Skip missing/disambiguation pages silently

    total_tokens_est = len(all_chunks) * 256
    print(f"[Knowledge] Fetched {fetched} articles → {len(all_chunks)} chunks (~{total_tokens_est:,} tokens estimated).")
    return all_chunks


# ── HuggingFace parquet loader (datasets v4+) ─────────────────────────────────

def _load_via_hf_parquet(n_articles: int = 600) -> list:
    """Load Wikipedia via direct Parquet file from HuggingFace Hub (datasets v4)."""
    from datasets import load_dataset
    # wikimedia/wikipedia has Parquet files — load first shard
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/wikimedia/wikipedia/20220301.en/train-00000-of-00041.parquet",
        split=f"train[:{n_articles}]",
    )
    all_chunks = []
    for row in ds:
        text = row.get("text", "") or ""
        if len(text) < 100:
            continue
        all_chunks.extend(_chunk_text(text, max_tokens=256)[:8])
    print(f"[Knowledge] HF Parquet: {len(ds)} articles → {len(all_chunks)} chunks.")
    return all_chunks


# ── Main loader ────────────────────────────────────────────────────────────────

def load_wikipedia_dataset(n_articles: int = 600, force_reload: bool = False) -> dict:
    """
    Load Wikipedia articles, chunk into 256-token chunks, extract entities.
    Tries: (1) cache → (2) wikipedia package → (3) HF parquet → (4) fallback KB.
    """
    # ── Check cache ────────────────────────────────────────────────────────────
    if not force_reload and os.path.exists(_CHUNKS_PKL) and os.path.exists(_ENTS_PKL):
        print("[Knowledge] Loading Wikipedia chunks from cache ...")
        with open(_CHUNKS_PKL, "rb") as f:
            chunks = pickle.load(f)
        with open(_ENTS_PKL, "rb") as f:
            ent_data = pickle.load(f)
        print(f"[Knowledge] Cache: {len(chunks)} chunks, {len(ent_data['entities'])} entities.")
        return {
            "chunks": chunks, "entities": ent_data["entities"],
            "relationships": ent_data["relationships"], "source": "cache",
        }

    # ── Fallback Knowledge Base ───────────────────────────────────────────────────
# ── ENTITIES ──────────────────────────────────────────────────────────────────
    ENTITIES = [
        # ── Machine Learning Core ─────────────────────────────────────────────────
        {
            "id": "ml_001",
            "name": "Machine Learning",
            "type": "Concept",
            "description": (
                "Machine Learning (ML) is a subset of artificial intelligence where "
                "systems learn patterns from data and improve their performance over time "
                "without being explicitly programmed. It uses statistical techniques to "
                "enable computers to learn from experience."
            ),
            "domain": "AI/ML",
        },
        {
            "id": "ml_002",
            "name": "Supervised Learning",
            "type": "Concept",
            "description": (
                "Supervised Learning is a type of machine learning where the model is "
                "trained on labeled data — each training example has an input and a known "
                "correct output. Examples: classification (spam detection) and regression "
                "(house price prediction)."
            ),
            "domain": "AI/ML",
        },
        {
            "id": "ml_003",
            "name": "Unsupervised Learning",
            "type": "Concept",
            "description": (
                "Unsupervised Learning trains models on data without labels. The model "
                "discovers hidden patterns or structure on its own. Common techniques include "
                "clustering (K-Means, DBSCAN) and dimensionality reduction (PCA, t-SNE)."
            ),
            "domain": "AI/ML",
        },
        {
            "id": "ml_004",
            "name": "Reinforcement Learning",
            "type": "Concept",
            "description": (
                "Reinforcement Learning (RL) trains an agent to make decisions by rewarding "
                "desired behaviors. The agent interacts with an environment, receives rewards "
                "or penalties, and learns optimal strategies. Used in game AI (AlphaGo), "
                "robotics, and autonomous vehicles."
            ),
            "domain": "AI/ML",
        },
        {
            "id": "ml_005",
            "name": "Feature Engineering",
            "type": "Concept",
            "description": (
                "Feature Engineering is the process of selecting, transforming, and creating "
                "input variables (features) to improve machine learning model performance. "
                "Techniques include normalization, one-hot encoding, and polynomial features."
            ),
            "domain": "AI/ML",
        },
        # ── Deep Learning ─────────────────────────────────────────────────────────
        {
            "id": "dl_001",
            "name": "Deep Learning",
            "type": "Concept",
            "description": (
                "Deep Learning is a subset of machine learning using artificial neural networks "
                "with many layers (deep networks). It excels at processing unstructured data "
                "like images, audio, and text. Powers modern AI breakthroughs in computer "
                "vision, speech recognition, and natural language processing."
            ),
            "domain": "AI/ML",
        },
        {
            "id": "dl_002",
            "name": "Neural Network",
            "type": "Concept",
            "description": (
                "A Neural Network is a computational model inspired by the human brain. It "
                "consists of layers of interconnected nodes (neurons) that process information. "
                "Trained using backpropagation and gradient descent."
            ),
            "domain": "AI/ML",
        },
        {
            "id": "dl_003",
            "name": "Convolutional Neural Network",
            "type": "Algorithm",
            "description": (
                "CNNs are specialized neural networks for processing grid-like data such as "
                "images. They use convolutional layers to detect local patterns. Famous "
                "architectures: ResNet, VGG, EfficientNet."
            ),
            "domain": "AI/ML",
        },
        {
            "id": "dl_004",
            "name": "Transformer Architecture",
            "type": "Algorithm",
            "description": (
                "The Transformer is a deep learning architecture based entirely on attention "
                "mechanisms, introduced in 'Attention Is All You Need' (2017). Foundation of "
                "all modern LLMs including BERT, GPT, and Groq-hosted models."
            ),
            "domain": "AI/ML",
        },
        {
            "id": "dl_005",
            "name": "Backpropagation",
            "type": "Algorithm",
            "description": (
                "Backpropagation is the algorithm used to train neural networks. It calculates "
                "gradients of the loss function using the chain rule, then uses gradient descent "
                "to update weights and minimize prediction errors."
            ),
            "domain": "AI/ML",
        },
        # ── NLP ───────────────────────────────────────────────────────────────────
        {
            "id": "nlp_001",
            "name": "Natural Language Processing",
            "type": "Concept",
            "description": (
                "NLP enables computers to understand, interpret, and generate human language. "
                "Applications include sentiment analysis, machine translation, chatbots, "
                "text summarization, and question answering."
            ),
            "domain": "NLP",
        },
        {
            "id": "nlp_002",
            "name": "Large Language Model",
            "type": "Concept",
            "description": (
                "LLMs are AI models trained on massive text datasets with billions of "
                "parameters. Examples: GPT-4, Groq LLaMA, Claude."
            ),
            "domain": "NLP",
        },
        {
            "id": "nlp_003",
            "name": "Retrieval-Augmented Generation",
            "type": "Concept",
            "description": (
                "RAG combines a retrieval system with an LLM. Relevant documents are first "
                "retrieved from a knowledge base, then passed as context to the LLM to generate "
                "a grounded, accurate answer. Reduces hallucinations."
            ),
            "domain": "NLP",
        },
        {
            "id": "nlp_004",
            "name": "Tokenization",
            "type": "Concept",
            "description": (
                "Tokenization splits text into smaller units called tokens. 1 token ≈ 4 "
                "characters or 0.75 words. Token count determines API cost and context window usage."
            ),
            "domain": "NLP",
        },
        {
            "id": "nlp_005",
            "name": "Embeddings",
            "type": "Concept",
            "description": (
                "Embeddings are dense numerical vector representations of text. Similar meanings "
                "map to nearby vectors in high-dimensional space. Used for semantic search."
            ),
            "domain": "NLP",
        },
        # ── Graph Technology ──────────────────────────────────────────────────────
        {
            "id": "graph_001",
            "name": "Graph Database",
            "type": "Tool",
            "description": (
                "A graph database stores data as nodes and edges. Excellent for highly connected "
                "data. Examples: TigerGraph, Neo4j, Amazon Neptune."
            ),
            "domain": "Databases",
        },
        {
            "id": "graph_002",
            "name": "TigerGraph",
            "type": "Tool",
            "description": (
                "TigerGraph is a high-performance native parallel graph database platform using "
                "GSQL. Excels at real-time deep link analytics across billions of edges."
            ),
            "domain": "Databases",
        },
        {
            "id": "graph_003",
            "name": "Knowledge Graph",
            "type": "Concept",
            "description": (
                "A Knowledge Graph is a structured representation of real-world entities and "
                "their relationships. Used by Google Search, Siri, and enterprise AI systems."
            ),
            "domain": "Databases",
        },
        {
            "id": "graph_004",
            "name": "GraphRAG",
            "type": "Concept",
            "description": (
                "GraphRAG extends RAG by using a knowledge graph as the retrieval source. "
                "Instead of searching flat documents, it traverses graph relationships to "
                "find multi-hop connected context for richer LLM answers."
            ),
            "domain": "Databases",
        },
        {
            "id": "graph_005",
            "name": "GSQL",
            "type": "Tool",
            "description": (
                "GSQL is TigerGraph's graph query language. Turing-complete, supporting "
                "pattern matching, multi-hop traversals, aggregations, and parallel execution."
            ),
            "domain": "Databases",
        },
        # ── Frameworks & Tools ────────────────────────────────────────────────────
        {
            "id": "fw_001",
            "name": "TensorFlow",
            "type": "Framework",
            "description": (
                "TensorFlow is an open-source ML framework by Google supporting building and "
                "training neural networks at scale with Keras API."
            ),
            "domain": "Tools",
        },
        {
            "id": "fw_002",
            "name": "PyTorch",
            "type": "Framework",
            "description": (
                "PyTorch is an open-source deep learning framework by Meta AI with dynamic "
                "computation graph. Dominant in research. Foundation of Hugging Face Transformers."
            ),
            "domain": "Tools",
        },
        {
            "id": "fw_003",
            "name": "Scikit-learn",
            "type": "Framework",
            "description": (
                "Scikit-learn is the most popular Python library for classical ML providing "
                "classification, regression, clustering, and preprocessing algorithms."
            ),
            "domain": "Tools",
        },
        {
            "id": "fw_004",
            "name": "Streamlit",
            "type": "Framework",
            "description": (
                "Streamlit is an open-source Python framework for building interactive web apps "
                "for data science and ML with pure Python code."
            ),
            "domain": "Tools",
        },
        {
            "id": "fw_005",
            "name": "Groq API",
            "type": "Tool",
            "description": (
                "Groq API provides fast access to state-of-the-art AI models like LLaMA 3, "
                "optimized for extreme speed and cost efficiency."
            ),
            "domain": "Tools",
        },
    ]
    
    
    # ── RELATIONSHIPS ──────────────────────────────────────────────────────────────
    RELATIONSHIPS = [
        {"source": "ml_001", "target": "dl_001",    "relation": "INCLUDES",        "description": "Machine Learning includes Deep Learning"},
        {"source": "ml_001", "target": "ml_002",    "relation": "INCLUDES",        "description": "Machine Learning includes Supervised Learning"},
        {"source": "ml_001", "target": "ml_003",    "relation": "INCLUDES",        "description": "Machine Learning includes Unsupervised Learning"},
        {"source": "ml_001", "target": "ml_004",    "relation": "INCLUDES",        "description": "Machine Learning includes Reinforcement Learning"},
        {"source": "ml_001", "target": "ml_005",    "relation": "REQUIRES",        "description": "Machine Learning requires Feature Engineering"},
        {"source": "dl_001", "target": "dl_002",    "relation": "USES",            "description": "Deep Learning uses Neural Networks"},
        {"source": "dl_001", "target": "dl_003",    "relation": "INCLUDES",        "description": "Deep Learning includes CNNs"},
        {"source": "dl_001", "target": "dl_004",    "relation": "USES",            "description": "Deep Learning uses Transformer Architecture"},
        {"source": "dl_002", "target": "dl_005",    "relation": "TRAINED_WITH",    "description": "Neural Networks are trained with Backpropagation"},
        {"source": "nlp_001","target": "dl_001",    "relation": "USES",            "description": "NLP uses Deep Learning"},
        {"source": "nlp_001","target": "nlp_002",   "relation": "USES",            "description": "NLP uses Large Language Models"},
        {"source": "nlp_002","target": "dl_004",    "relation": "BUILT_ON",        "description": "LLMs are built on the Transformer Architecture"},
        {"source": "nlp_002","target": "nlp_004",   "relation": "USES",            "description": "LLMs use Tokenization"},
        {"source": "nlp_003","target": "nlp_002",   "relation": "ENHANCES",        "description": "RAG enhances LLMs with retrieved context"},
        {"source": "nlp_003","target": "graph_004", "relation": "EXTENDED_BY",     "description": "Traditional RAG is extended by GraphRAG"},
        {"source": "nlp_005","target": "nlp_001",   "relation": "ENABLES",         "description": "Embeddings enable semantic NLP tasks"},
        {"source": "graph_004","target": "graph_001","relation": "USES",           "description": "GraphRAG uses a Graph Database"},
        {"source": "graph_004","target": "graph_003","relation": "USES",           "description": "GraphRAG uses Knowledge Graphs"},
        {"source": "graph_002","target": "graph_001","relation": "IS_A",           "description": "TigerGraph is a type of Graph Database"},
        {"source": "graph_002","target": "graph_005","relation": "USES",           "description": "TigerGraph uses GSQL"},
        {"source": "graph_003","target": "nlp_005", "relation": "USES",            "description": "Knowledge Graphs use Embeddings"},
        {"source": "fw_001",  "target": "ml_001",   "relation": "IMPLEMENTS",      "description": "TensorFlow implements Machine Learning"},
        {"source": "fw_002",  "target": "dl_001",   "relation": "IMPLEMENTS",      "description": "PyTorch implements Deep Learning"},
        {"source": "fw_003",  "target": "ml_002",   "relation": "IMPLEMENTS",      "description": "Scikit-learn implements Supervised Learning"},
        {"source": "fw_005",  "target": "nlp_002",  "relation": "PROVIDES_ACCESS", "description": "Groq API provides access to LLMs"},
        {"source": "fw_004",  "target": "graph_004","relation": "USED_FOR",        "description": "Streamlit builds GraphRAG dashboards"},
    ]
    
    
    # ── CHUNKS (plain text for FAISS) ─────────────────────────────────────────────
    CHUNKS = [
        "Machine Learning (ML) is a subset of artificial intelligence where systems learn patterns "
        "from data and improve their performance over time without being explicitly programmed. "
        "It uses statistical techniques to enable computers to learn from experience.",
    
        "Supervised Learning is a type of machine learning where the model is trained on labeled data. "
        "Each training example has an input and a known correct output. "
        "Examples include classification tasks like spam detection and regression tasks like house price prediction.",
    
        "Unsupervised Learning trains models on data without labels, allowing the model to discover "
        "hidden patterns or structure on its own. Common techniques include clustering algorithms "
        "such as K-Means and DBSCAN, and dimensionality reduction methods like PCA and t-SNE.",
    
        "Reinforcement Learning (RL) trains an agent to make decisions by rewarding desired behaviors. "
        "The agent interacts with an environment, receives rewards or penalties, and learns optimal strategies. "
        "It is used in game AI (AlphaGo), robotics, and autonomous vehicles.",
    
        "Feature Engineering is the process of selecting, transforming, and creating input variables (features) "
        "to improve machine learning model performance. Good features help models learn patterns more effectively. "
        "Techniques include normalization, one-hot encoding, and polynomial features.",
    
        "Deep Learning is a subset of machine learning using artificial neural networks with many layers, "
        "called deep networks. It excels at processing unstructured data like images, audio, and text. "
        "It powers modern AI breakthroughs in computer vision, speech recognition, and natural language processing.",
    
        "A Neural Network is a computational model inspired by the human brain. It consists of layers of "
        "interconnected nodes (neurons) that process information. The input layer receives data, hidden layers "
        "transform it, and the output layer produces predictions. Neural networks are trained using backpropagation.",
    
        "Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data "
        "such as images. They use convolutional layers to detect local patterns like edges, textures, and shapes. "
        "Famous architectures include ResNet, VGG, and EfficientNet.",
    
        "The Transformer is a deep learning architecture based entirely on attention mechanisms, introduced in "
        "'Attention Is All You Need' (2017). It processes sequences in parallel, unlike RNNs, making training faster. "
        "The Transformer is the foundation of all modern LLMs including BERT, GPT, and Groq-hosted models.",
    
        "Backpropagation is the algorithm used to train neural networks. It calculates the gradient of the loss "
        "function with respect to each weight by applying the chain rule of calculus. It then uses gradient descent "
        "to update the weights and minimize prediction errors iteratively.",
    
        "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. "
        "Applications include sentiment analysis, machine translation, chatbots, text summarization, and question answering.",
    
        "Large Language Models (LLMs) are AI models trained on massive text datasets with billions of parameters. "
        "They can generate, summarize, translate, and answer questions in natural language. "
        "Examples include GPT-4, Groq LLaMA, and Claude.",
    
        "Retrieval-Augmented Generation (RAG) combines a retrieval system with an LLM. When a user asks a question, "
        "relevant documents are first retrieved from a knowledge base and passed as context to the LLM. "
        "RAG reduces hallucinations and allows LLMs to access up-to-date information.",
    
        "Tokenization splits text into smaller units called tokens. In modern LLMs, tokens are sub-word pieces. "
        "A rough rule is that 1 token equals approximately 4 characters or 0.75 words. "
        "Token count determines API cost and context window usage.",
    
        "Embeddings are dense numerical vector representations of text or other data. Similar meanings map to "
        "nearby vectors in high-dimensional space. They are used for semantic search, recommendation systems, "
        "and as input to machine learning models.",
    
        "A Graph Database stores data as nodes (entities) and edges (relationships) rather than tables. "
        "It is excellent for highly connected data and supports multi-hop traversal queries. "
        "Examples include TigerGraph, Neo4j, and Amazon Neptune.",
    
        "TigerGraph is a high-performance native parallel graph database platform. It uses GSQL, a SQL-like "
        "graph query language, and excels at real-time deep link analytics. TigerGraph can perform multi-hop "
        "traversals across billions of edges in seconds.",
    
        "A Knowledge Graph is a structured representation of real-world entities and their relationships. "
        "It organizes information semantically so machines can reason about it. "
        "Knowledge Graphs are used by Google Search, Siri, and enterprise AI systems.",
    
        "GraphRAG extends RAG by using a knowledge graph as the retrieval source. Instead of searching flat "
        "documents, it traverses graph relationships to find multi-hop connected context. "
        "This provides richer, more structured context to LLMs, improving answer accuracy for complex questions.",
    
        "GSQL is TigerGraph's graph query language. It is Turing-complete and supports pattern matching, "
        "multi-hop traversals, aggregations, and updates on graph data. Supports parallel execution.",
    
        "TensorFlow is an open-source ML framework developed by Google. It supports building and training neural "
        "networks at scale. Features include the Keras high-level API for rapid prototyping.",
    
        "PyTorch is an open-source deep learning framework developed by Meta AI. It is known for its dynamic "
        "computation graph (eager execution), making debugging intuitive. PyTorch is dominant in research.",
    
        "Scikit-learn is the most popular Python library for classical machine learning. It provides a simple, "
        "consistent API for classification, regression, clustering, and preprocessing.",
    
        "Streamlit is an open-source Python framework for building interactive web apps for data science and ML. "
        "You write pure Python and get a shareable web app.",
    
        "Groq API provides fast access to state-of-the-art AI models like LLaMA 3. Groq is optimized for "
        "extreme speed and cost efficiency.",
    
        "Vector databases store embeddings (dense numerical vectors) and allow similarity search. FAISS (Facebook "
        "AI Similarity Search) is a popular vector library for efficient nearest-neighbor search. "
        "Vector databases are the backbone of Basic RAG systems.",
    
        "RAG vs GraphRAG: Basic RAG retrieves flat document chunks using vector similarity search with FAISS. "
        "GraphRAG retrieves structured context by traversing a knowledge graph using multi-hop queries. "
        "GraphRAG provides richer context for complex questions that require connecting multiple facts.",
    
        "The hackathon system uses three pipelines: Pipeline 1 is LLM-Only (direct question to Groq), Pipeline 2 "
        "is Basic RAG (FAISS vector search + Groq), and Pipeline 3 is GraphRAG (TigerGraph multi-hop + Groq). "
        "This allows comparison of retrieval strategies and their impact on answer quality.",
    ]
    
    
    
    _HE = ENTITIES
    _HR = RELATIONSHIPS
    _HC = CHUNKS


    wiki_chunks = []

    # ── Try (1): wikipedia Python package ─────────────────────────────────────
    try:
        wiki_chunks = _load_via_wikipedia_package(n_articles)
    except Exception as e1:
        print(f"[Knowledge] wikipedia package failed: {e1}")

    # ── Try (2): HuggingFace parquet ──────────────────────────────────────────
    if not wiki_chunks:
        try:
            wiki_chunks = _load_via_hf_parquet(n_articles)
        except Exception as e2:
            print(f"[Knowledge] HF parquet failed: {e2}")

    # ── Fallback ───────────────────────────────────────────────────────────────
    if not wiki_chunks:
        print("[Knowledge] All Wikipedia methods failed — using built-in KB only.")
        return {"chunks": _HC, "entities": _HE, "relationships": _HR, "source": "fallback"}

    # Merge: hardcoded KB chunks first (high quality), then Wikipedia
    all_chunks = _HC + wiki_chunks

    print(f"[Knowledge] Extracting entities from {len(all_chunks)} chunks ...")
    wiki_ents, wiki_rels = _extract_entities_from_chunks(wiki_chunks)
    all_entities  = _HE + wiki_ents
    all_rels      = _HR + wiki_rels

    total_tokens_est = len(all_chunks) * 256
    print(f"[Knowledge] Total: {len(all_chunks)} chunks (~{total_tokens_est:,} tokens), {len(all_entities)} entities, {len(all_rels)} rels.")

    # ── Save cache ─────────────────────────────────────────────────────────────
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CHUNKS_PKL, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(_ENTS_PKL, "wb") as f:
        pickle.dump({"entities": all_entities, "relationships": all_rels}, f)
    print(f"[Knowledge] Cache saved.")

    return {
        "chunks": all_chunks, "entities": all_entities,
        "relationships": all_rels, "source": "wikipedia",
    }


# ── Singleton ──────────────────────────────────────────────────────────────────

_KB: "dict|None" = None


def get_knowledge_base(force_reload: bool = False) -> dict:
    global _KB
    if _KB is None or force_reload:
        # 600 topics × up to 20 chunks × 256 tokens ≈ 3M tokens from Wikipedia.
        _KB = load_wikipedia_dataset(n_articles=600, force_reload=force_reload)
    return _KB


def get_chunks() -> list:
    return get_knowledge_base()["chunks"]


def get_all_entities() -> list:
    return get_knowledge_base()["entities"]


def get_all_relationships() -> list:
    return get_knowledge_base()["relationships"]


def get_entity_by_id(entity_id: str):
    for e in get_all_entities():
        if e["id"] == entity_id:
            return e
    return None


# ── Module-level exports (for tests and imports) ───────────────────────────────
# These are populated lazily on first import access.
class _LazyList(list):
    """List that loads the KB on first access."""
    def __init__(self, key):
        super().__init__()
        self._key = key
        self._loaded = False
    def _ensure(self):
        if not self._loaded:
            data = get_knowledge_base()
            self.extend(data[self._key])
            self._loaded = True
    def __len__(self): self._ensure(); return super().__len__()
    def __getitem__(self, i): self._ensure(); return super().__getitem__(i)
    def __iter__(self): self._ensure(); return super().__iter__()
    def __repr__(self): self._ensure(); return super().__repr__()


CHUNKS        = _LazyList("chunks")
ENTITIES      = _LazyList("entities")
RELATIONSHIPS = _LazyList("relationships")

# ── Ground Truth ──
GROUND_TRUTH = [
    {"question": "What is machine learning?",
     "correct_answer": "Machine learning is a subset of AI that enables systems to learn from data and improve automatically without explicit programming. It includes supervised learning with labeled data, unsupervised learning for pattern discovery, and reinforcement learning through rewards. Applications include image recognition, spam filtering, and recommendation systems."},
    {"question": "What is deep learning?",
     "correct_answer": "Deep learning uses neural networks with many layers to automatically learn features from raw data like images and text. It is a subset of machine learning that requires large datasets and GPU computation. Deep learning powers image recognition, speech recognition, and large language models."},
    {"question": "What is a graph database?",
     "correct_answer": "A graph database stores data as nodes and edges to represent and query complex relationships efficiently. Unlike relational databases, graph databases support multi-hop traversal queries across connected data. TigerGraph, Neo4j, and Amazon Neptune are popular examples."},
    {"question": "What is natural language processing?",
     "correct_answer": "Natural language processing enables computers to understand, interpret, and generate human language using machine learning. It powers applications like chatbots, machine translation, sentiment analysis, and text summarization. Key techniques include tokenization, named entity recognition, and transformer-based language models."},
    {"question": "What is RAG in AI?",
     "correct_answer": "Retrieval Augmented Generation enhances language model responses by retrieving relevant documents from a knowledge base before generating an answer. RAG reduces hallucination and improves factual accuracy beyond the model's training cutoff. It is widely used in enterprise question answering and document search systems."},
    {"question": "What is TigerGraph?",
     "correct_answer": "TigerGraph is a native parallel graph database designed for enterprise-scale analytics using the GSQL query language. It excels at real-time deep link analytics including fraud detection, recommendation engines, and knowledge graph traversal. Its distributed architecture processes billions of vertices and edges with fast query response times."},
    {"question": "What is GraphRAG?",
     "correct_answer": "GraphRAG is an advanced retrieval approach that uses a knowledge graph instead of flat vector search to provide context for language models. It traverses graph relationships to find multi-hop connected entities, providing richer context than basic RAG. GraphRAG produces more accurate answers while using fewer tokens than traditional RAG systems."},
    {"question": "What is a neural network?",
     "correct_answer": "A neural network is a computational model inspired by the brain consisting of interconnected layers of artificial neurons that process information. Networks learn by adjusting connection weights through backpropagation to minimize prediction errors. They are the foundation of deep learning used for image classification, speech recognition, and language translation."},
    {"question": "What is transfer learning?",
     "correct_answer": "Transfer learning reuses a model trained on one task as the starting point for a different but related task rather than training from scratch. It leverages knowledge from large pretrained models like BERT and GPT that can be fine-tuned for specific downstream tasks. Transfer learning dramatically reduces training time and data requirements while achieving strong performance."},
    {"question": "What is the transformer architecture?",
     "correct_answer": "The transformer architecture is a deep learning model that uses self-attention mechanisms to process input sequences in parallel instead of sequentially. It consists of encoder and decoder blocks with multi-head attention and feed-forward layers. Transformers are the foundation of modern language models like GPT, BERT, and LLaMA."},
    {"question": "What is FAISS?",
     "correct_answer": "FAISS is an open-source library by Facebook AI Research for efficient similarity search and clustering of dense vectors at billion scale. It implements approximate nearest neighbor search algorithms for fast retrieval in large vector collections. FAISS is a key component in basic RAG systems for semantic document retrieval."},
    {"question": "What are embeddings?",
     "correct_answer": "Embeddings are dense numerical vector representations of text where similar meanings are located close together in vector space. They allow computers to perform mathematical operations on language and calculate semantic similarity. Embeddings are generated by neural networks and power semantic search and modern NLP applications."},
    {"question": "What is a vector database?",
     "correct_answer": "A vector database stores and indexes high-dimensional vector embeddings for efficient similarity search using approximate nearest neighbor algorithms. It quickly finds the most similar vectors to a query from millions of stored embeddings. Vector databases like Pinecone, Milvus, and Qdrant are essential for RAG applications."},
    {"question": "What is supervised learning?",
     "correct_answer": "Supervised learning trains machine learning models on labeled datasets where each input has a known correct output target. The model learns to predict outputs for new data by minimizing errors on the training labels. Common tasks include classification for spam detection and regression for predicting house prices."},
    {"question": "What is unsupervised learning?",
     "correct_answer": "Unsupervised learning trains models on unlabeled data to discover hidden patterns or structures without predefined targets. Common techniques include clustering algorithms like K-means and dimensionality reduction methods like PCA. It is used for customer segmentation, anomaly detection, and exploratory data analysis."},
    {"question": "What is reinforcement learning?",
     "correct_answer": "Reinforcement learning trains an agent to make decisions by interacting with an environment to maximize cumulative rewards. The agent learns through trial and error, receiving positive rewards or penalties based on its actions. This technique achieves superhuman performance in games like Go and is used in robotics and autonomous systems."},
    {"question": "What is a convolutional neural network?",
     "correct_answer": "A convolutional neural network is a deep learning architecture that processes grid-structured data like images using convolutional layers with learnable filters. These filters automatically detect spatial hierarchies of features from edges to complex objects through weight sharing. CNNs are the standard approach for image classification and object detection tasks."},
    {"question": "What is a recurrent neural network?",
     "correct_answer": "A recurrent neural network processes sequential data like text and time series by maintaining an internal memory state across sequence steps. This enables the network to understand dependencies over time in sequential input. Advanced variants like LSTMs and GRUs address the vanishing gradient problem for long sequences."},
    {"question": "What is BERT?",
     "correct_answer": "BERT is a transformer-based language model from Google that processes text bidirectionally to understand word context from both sides. It is pretrained on large corpora using masked language modeling and can be fine-tuned for downstream NLP tasks. BERT powers question answering, sentiment analysis, and named entity recognition applications."},
    {"question": "What is GPT?",
     "correct_answer": "GPT is a family of large language models by OpenAI using a decoder-only transformer architecture pretrained to predict the next token in text. GPT models exhibit strong zero-shot and few-shot learning capabilities across diverse language tasks. They are widely used for text generation, translation, coding assistance, and conversational AI."},
    {"question": "What is tokenization?",
     "correct_answer": "Tokenization splits raw text into smaller units called tokens that machine learning models can process mathematically. Modern LLMs use subword tokenization like Byte Pair Encoding to handle rare words with a manageable vocabulary. The tokenizer converts tokens into numerical IDs that the neural network ingests as input."},
    {"question": "What is overfitting?",
     "correct_answer": "Overfitting occurs when a machine learning model learns the training data too closely, capturing noise rather than the true underlying pattern. An overfitted model performs well on training data but fails to generalize to new unseen examples. Techniques like cross-validation, dropout, and regularization are used to prevent overfitting."},
    {"question": "What is regularization?",
     "correct_answer": "Regularization techniques prevent machine learning models from overfitting by adding a penalty to the loss function that discourages overly complex parameter weights. Common methods include L1 and L2 regularization, dropout in neural networks, and early stopping. Regularization forces models to learn simpler and more generalizable representations."},
    {"question": "What is clustering?",
     "correct_answer": "Clustering is an unsupervised machine learning technique that groups similar data points together based on their features without predefined labels. K-means and hierarchical clustering are widely used algorithms for this purpose. Clustering is applied for customer segmentation, image compression, and exploratory data analysis."},
    {"question": "What is classification?",
     "correct_answer": "Classification is a supervised machine learning task that predicts the categorical class label of an input data point based on learned patterns. Models are trained on labeled examples and learn decision boundaries to separate different categories. Common applications include spam email detection, medical diagnosis, and image recognition."},
    {"question": "What is regression?",
     "correct_answer": "Regression is a supervised machine learning task focused on predicting a continuous numerical output variable from input features. Linear regression is the simplest form, learning the relationship between variables from historical training data. Regression is used for forecasting stock prices, housing values, and other continuous quantities."},
    {"question": "What is a knowledge graph?",
     "correct_answer": "A knowledge graph represents real-world entities and their relationships as a structured network of nodes and edges. This graph structure enables machines to understand context and perform logical reasoning across interconnected domains. Knowledge graphs power Google Search, recommendation systems, and enterprise AI applications."},
    {"question": "What is entity extraction?",
     "correct_answer": "Entity extraction, also called named entity recognition, automatically identifies and classifies key entities in text such as person names, organizations, locations, and dates. It transforms unstructured text into structured data that can be queried and analyzed. Entity extraction is a critical step in building knowledge graphs and information retrieval systems."},
    {"question": "What is semantic search?",
     "correct_answer": "Semantic search finds documents by understanding the meaning and intent behind a query rather than just matching keywords. It uses vector embeddings and similarity search to find conceptually related content even with different vocabulary. Semantic search improves relevance over keyword search and is the foundation of RAG systems."},
    {"question": "What is cosine similarity?",
     "correct_answer": "Cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them, ranging from 0 to 1. In NLP it is the standard method for comparing text embeddings to determine how semantically similar two texts are. Cosine similarity is efficient and robust to differences in document length."},
]

