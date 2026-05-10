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
    {
        "question": "What is machine learning?",
        "correct_answer": "Machine learning is a branch of artificial intelligence that enables computer systems to automatically learn and improve from experience without being explicitly programmed. It works by training algorithms on large datasets to identify patterns and make data-driven decisions. The three main types are supervised learning which uses labeled data, unsupervised learning which finds hidden patterns, and reinforcement learning which learns through rewards. Common applications include image recognition, spam filtering, recommendation systems, and natural language processing."
    },
    {
        "question": "What is deep learning?",
        "correct_answer": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to learn representations of data at increasing levels of abstraction. These deep neural networks are inspired by the structure of the human brain and can automatically discover features from raw data without manual feature engineering. Deep learning excels at tasks like image recognition, speech recognition, natural language processing, and game playing. It requires large amounts of training data and significant computational resources typically provided by GPUs."
    },
    {
        "question": "What is a graph database?",
        "correct_answer": "A graph database is a type of NoSQL database that uses graph structures with nodes, edges, and properties to store and represent data. Unlike relational databases that use tables, graph databases excel at handling complex relationships between data points and performing multi-hop queries that traverse multiple connections. They are particularly useful for social networks, recommendation engines, fraud detection, and knowledge graphs. Popular graph databases include TigerGraph, Neo4j, and Amazon Neptune."
    },
    {
        "question": "What is natural language processing?",
        "correct_answer": "Natural language processing is a branch of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language in a meaningful way. It combines computational linguistics with machine learning and deep learning to process and analyze large amounts of natural language data. NLP powers many everyday applications including virtual assistants, machine translation, sentiment analysis, and chatbots. Key techniques include tokenization, part-of-speech tagging, named entity recognition, and transformer-based language models."
    },
    {
        "question": "What is RAG in AI?",
        "correct_answer": "Retrieval Augmented Generation is an AI framework that enhances large language model responses by retrieving relevant information from external knowledge sources before generating an answer. Instead of relying solely on knowledge encoded during training, RAG systems first search a document database or knowledge base to find relevant context and then use that context to generate more accurate and up-to-date responses. This approach reduces hallucination, improves factual accuracy, and allows the model to access information beyond its training cutoff. RAG is widely used in enterprise question answering systems."
    },
    {
        "question": "What is TigerGraph?",
        "correct_answer": "TigerGraph is a native parallel graph database and analytics platform designed for enterprise-scale graph analytics and machine learning. It uses a proprietary query language called GSQL which allows users to perform complex multi-hop graph traversals and pattern matching at high speed. TigerGraph excels at real-time deep link analytics, fraud detection, recommendation engines, and knowledge graph applications. Its distributed architecture allows it to process graphs with hundreds of billions of vertices and edges while maintaining fast query response times."
    },
    {
        "question": "What is GraphRAG?",
        "correct_answer": "GraphRAG is an advanced retrieval augmented generation approach that uses knowledge graphs instead of or in addition to vector databases for context retrieval. Unlike traditional RAG which retrieves similar text chunks using vector embeddings, GraphRAG traverses graph relationships to find connected entities and their relationships through multi-hop reasoning. This allows GraphRAG to answer complex questions requiring synthesis of information from multiple connected documents or entities. GraphRAG typically produces more accurate and contextually rich answers while using fewer tokens than basic RAG."
    },
    {
        "question": "What is a neural network?",
        "correct_answer": "A neural network is a computational model inspired by the structure and function of biological neural networks in the human brain. It consists of interconnected layers of artificial neurons or nodes that process information by passing signals through weighted connections. Neural networks learn by adjusting these weights through a process called backpropagation which minimizes prediction errors on training data. They are the foundation of modern deep learning and are used for tasks including image classification, speech recognition, language translation, and game playing."
    },
    {
        "question": "What is transfer learning?",
        "correct_answer": "Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different but related task. Instead of training a model from scratch which requires large amounts of data and computation, transfer learning allows practitioners to leverage knowledge already learned from large datasets. This approach is particularly powerful in deep learning where pretrained models like BERT and GPT are fine-tuned on specific downstream tasks. Transfer learning dramatically reduces training time and data requirements while often achieving better performance."
    },
    {
        "question": "What is the transformer architecture?",
        "correct_answer": "The transformer architecture is a deep learning model architecture introduced in the paper Attention Is All You Need that relies entirely on self-attention mechanisms instead of recurrent or convolutional layers. It processes input sequences in parallel rather than sequentially making it highly efficient for training on modern hardware. The transformer consists of encoder and decoder blocks each containing multi-head attention layers and feed-forward networks. It has become the dominant architecture for natural language processing tasks and is the foundation of large language models like GPT, BERT, T5, and LLaMA."
    },
    {
        "question": "What is FAISS?",
        "correct_answer": "FAISS is an open-source library developed by Facebook AI Research that specializes in highly efficient similarity search and clustering of dense vectors. It is designed to scale to datasets of billions of vectors that may not fit entirely in RAM. FAISS implements various algorithms for approximate nearest neighbor search including quantization and hierarchical navigable small world graphs. It is a critical component in many retrieval augmented generation systems for fast document retrieval."
    },
    {
        "question": "What are embeddings?",
        "correct_answer": "Embeddings are dense numerical vector representations of data such as text, images, or audio in a high-dimensional continuous space. They capture semantic meaning and relationships where similar concepts are located close to each other in the vector space. Text embeddings allow computers to perform mathematical operations on language and calculate semantic similarity between words or sentences. They are typically generated by neural networks and serve as the foundation for semantic search and modern NLP."
    },
    {
        "question": "What is a vector database?",
        "correct_answer": "A vector database is a specialized storage system designed to efficiently store, index, and query high-dimensional vector embeddings generated by machine learning models. Unlike traditional relational databases, vector databases use approximate nearest neighbor algorithms to quickly find the vectors most similar to a given query vector. They are essential for semantic search, recommendation engines, and retrieval augmented generation applications. Popular vector databases include Pinecone, Milvus, Qdrant, and Weaviate."
    },
    {
        "question": "What is supervised learning?",
        "correct_answer": "Supervised learning is a core machine learning paradigm where an algorithm is trained on a labeled dataset containing input features and corresponding correct output targets. The model learns to map inputs to outputs by minimizing the error between its predictions and the actual labels during training. Once trained, the model can predict outcomes for new, unseen data. Common supervised learning tasks include classification such as spam detection and regression such as predicting house prices."
    },
    {
        "question": "What is unsupervised learning?",
        "correct_answer": "Unsupervised learning is a type of machine learning where models are trained on unlabeled data to discover hidden patterns, structures, or relationships without predefined target outputs. The algorithm explores the data autonomously to group similar instances or reduce the number of features while preserving essential information. Common techniques include clustering algorithms like K-means and dimensionality reduction methods like principal component analysis. It is widely used for customer segmentation and anomaly detection."
    },
    {
        "question": "What is reinforcement learning?",
        "correct_answer": "Reinforcement learning is a machine learning approach where an agent learns to make sequential decisions by interacting with an environment to maximize a cumulative reward. The agent takes actions and receives feedback in the form of positive rewards or negative penalties based on the outcome of its actions. Through trial and error, the agent develops an optimal policy that maps situations to the best possible actions. This technique has achieved superhuman performance in complex games like Go and is heavily used in robotics."
    },
    {
        "question": "What is a convolutional neural network?",
        "correct_answer": "A convolutional neural network is a specialized deep learning architecture designed primarily for processing structured grid data such as images and video. It uses convolutional layers with learnable filters to automatically detect spatial hierarchies of features ranging from simple edges to complex objects. CNNs significantly reduce the number of parameters compared to fully connected networks through weight sharing and pooling operations. They are the standard approach for computer vision tasks like image classification and object detection."
    },
    {
        "question": "What is a recurrent neural network?",
        "correct_answer": "A recurrent neural network is a type of artificial neural network designed to process sequential data such as time series, speech, or text. Unlike feedforward networks, RNNs have internal memory and loops that allow information to persist across different steps in the sequence. This architecture enables the network to maintain context and understand dependencies over time. While traditional RNNs suffer from vanishing gradient problems, advanced variants like LSTMs and GRUs are highly effective for sequence modeling tasks."
    },
    {
        "question": "What is BERT?",
        "correct_answer": "BERT stands for Bidirectional Encoder Representations from Transformers and is a landmark language model developed by Google. It processes text bidirectionally to deeply understand the context of words based on their surrounding text on both the left and the right. BERT is pretrained on massive text corpora using masked language modeling and next sentence prediction objectives. It can be easily fine-tuned for a wide variety of downstream natural language processing tasks such as question answering and sentiment analysis."
    },
    {
        "question": "What is GPT?",
        "correct_answer": "Generative Pre-trained Transformer is a family of large language models developed by OpenAI that use a decoder-only transformer architecture. GPT models are pretrained on vast amounts of internet text to predict the next token in a sequence, allowing them to learn deep patterns of human language and general knowledge. Due to their immense scale, recent GPT models exhibit remarkable zero-shot and few-shot learning capabilities. They are widely used for text generation, translation, coding, and conversational AI."
    },
    {
        "question": "What is tokenization?",
        "correct_answer": "Tokenization is the fundamental preprocessing step in natural language processing where raw text is divided into smaller units called tokens. These tokens can be words, characters, or subword pieces that the model can mathematically process. Modern large language models typically use subword tokenization algorithms like Byte Pair Encoding to effectively handle rare words and maintain a manageable vocabulary size. The tokenizer converts these tokens into numerical IDs that the neural network can ingest."
    },
    {
        "question": "What is overfitting?",
        "correct_answer": "Overfitting is a common modeling error in machine learning where a model learns the training data too well, capturing noise and random fluctuations rather than the underlying pattern. As a result, the overfitted model performs exceptionally well on the training data but fails to generalize and performs poorly on new, unseen test data. It typically occurs when a model is excessively complex relative to the amount of training data available. Techniques like cross-validation and regularization are used to prevent overfitting."
    },
    {
        "question": "What is regularization?",
        "correct_answer": "Regularization refers to a set of techniques used in machine learning to prevent models from overfitting the training data and improve their generalization to new data. These techniques typically involve adding a penalty term to the loss function that discourages the model from learning overly complex patterns or having large parameter weights. Common regularization methods include L1 and L2 regularization, dropout in neural networks, and early stopping. Regularization forces the model to learn simpler and more robust representations."
    },
    {
        "question": "What is clustering?",
        "correct_answer": "Clustering is a fundamental unsupervised machine learning technique used to group a set of unlabeled data points into distinct clusters based on their inherent similarities. The goal is to ensure that data points within the same cluster are highly similar to each other while being dissimilar to points in other clusters. K-means and hierarchical clustering are widely used algorithms for this task. Clustering is heavily utilized for customer segmentation, image compression, and exploratory data analysis."
    },
    {
        "question": "What is classification?",
        "correct_answer": "Classification is a primary supervised machine learning task where the goal is to predict the categorical class label of a given input data point based on past observations. The model is trained on a dataset containing instances with known labels and learns decision boundaries to separate different classes. It can be binary classification with two classes or multiclass classification with several categories. Common applications include email spam filtering, medical diagnosis, and image recognition."
    },
    {
        "question": "What is regression?",
        "correct_answer": "Regression is a core supervised machine learning task focused on predicting a continuous numerical output variable based on one or more input features. The algorithm learns the underlying mathematical relationship between the independent variables and the dependent target variable from historical training data. Linear regression is the simplest and most common form of this technique. Regression is widely applied in forecasting scenarios such as predicting stock prices, housing values, and temperature."
    },
    {
        "question": "What is a knowledge graph?",
        "correct_answer": "A knowledge graph is a structured data model that represents real-world entities and their interconnected relationships in a network format. It stores information as a collection of nodes representing concepts or objects and edges representing the semantic links between them. This graph-based structure allows machines to understand context and perform complex logical reasoning across interconnected domains. Knowledge graphs power major applications like Google Search, recommendation systems, and advanced enterprise AI."
    },
    {
        "question": "What is entity extraction?",
        "correct_answer": "Entity extraction, also known as named entity recognition, is an important natural language processing technique used to automatically locate and classify specific entities within unstructured text. The algorithm identifies proper nouns and key concepts and categorizes them into predefined classes such as person names, organizations, locations, and dates. This technique transforms raw text into structured data that can be easily queried and analyzed. It is a critical first step in building knowledge graphs and information retrieval systems."
    },
    {
        "question": "What is semantic search?",
        "correct_answer": "Semantic search is an advanced search technique that focuses on understanding the intent and contextual meaning behind a user query rather than relying strictly on exact keyword matching. It utilizes natural language processing and vector embeddings to find documents that are conceptually similar to the search query even if they use different vocabulary. This approach significantly improves search relevance and user experience by overcoming the limitations of traditional lexical search. Semantic search is the foundation of modern retrieval augmented generation."
    },
    {
        "question": "What is cosine similarity?",
        "correct_answer": "Cosine similarity is a mathematical metric used to measure the similarity between two non-zero vectors by calculating the cosine of the angle between them in a multidimensional space. The resulting value ranges from -1 to 1, where 1 indicates identical direction and 0 indicates orthogonality. In natural language processing, it is the standard method for determining how closely related two text documents or queries are by comparing their dense embedding vectors. Cosine similarity is highly efficient and robust to differences in document length."
    },
]

