"""
data/knowledge.py - Tech Knowledge Dataset
==========================================
This file contains the knowledge base that will be loaded
into TigerGraph as nodes (entities) and edges (relationships).

Beginner tip:
  - ENTITIES = things (nodes in the graph)
  - RELATIONSHIPS = connections between things (edges in the graph)

We've created a tech/AI knowledge base covering Machine Learning,
Deep Learning, NLP, Databases, and Cloud topics.
"""

# ── ENTITIES ──────────────────────────────────────────────────────────────────
# Each entity has:
#   id         : unique identifier (used as the graph node ID)
#   name       : human-readable name
#   type       : category (Concept, Algorithm, Tool, Framework, etc.)
#   description: a short explanation (this becomes the graph context for LLM)
#   domain     : broad topic area

ENTITIES = [
    # ── Machine Learning Core ────────────────────────────────────────────────
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
            "correct output. The model learns to map inputs to outputs. Examples: "
            "classification (spam detection) and regression (house price prediction)."
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
            "Good features help models learn patterns more effectively. Techniques include "
            "normalization, one-hot encoding, and polynomial features."
        ),
        "domain": "AI/ML",
    },

    # ── Deep Learning ────────────────────────────────────────────────────────
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
            "Input layer receives data, hidden layers transform it, output layer produces "
            "predictions. Trained using backpropagation and gradient descent."
        ),
        "domain": "AI/ML",
    },
    {
        "id": "dl_003",
        "name": "Convolutional Neural Network",
        "type": "Algorithm",
        "description": (
            "CNNs are specialized neural networks for processing grid-like data such as "
            "images. They use convolutional layers to detect local patterns (edges, textures, "
            "shapes). Widely used in image classification, object detection, and medical "
            "image analysis. Famous architectures: ResNet, VGG, EfficientNet."
        ),
        "domain": "AI/ML",
    },
    {
        "id": "dl_004",
        "name": "Transformer Architecture",
        "type": "Algorithm",
        "description": (
            "The Transformer is a deep learning architecture based entirely on attention "
            "mechanisms, introduced in 'Attention Is All You Need' (2017). It processes "
            "sequences in parallel (unlike RNNs) making training faster. Foundation of "
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
            "the gradient of the loss function with respect to each weight by applying the "
            "chain rule of calculus, then uses gradient descent to update the weights and "
            "minimize prediction errors."
        ),
        "domain": "AI/ML",
    },

    # ── NLP ──────────────────────────────────────────────────────────────────
    {
        "id": "nlp_001",
        "name": "Natural Language Processing",
        "type": "Concept",
        "description": (
            "NLP enables computers to understand, interpret, and generate human language. "
            "It bridges the gap between human communication and computer understanding. "
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
            "parameters. They can generate, summarize, translate, and answer questions "
            "in natural language. Examples: GPT-4, Groq LLaMA, Claude. Trained using "
            "self-supervised learning on internet-scale data."
        ),
        "domain": "NLP",
    },
    {
        "id": "nlp_003",
        "name": "Retrieval-Augmented Generation",
        "type": "Concept",
        "description": (
            "RAG combines a retrieval system with an LLM. When a user asks a question, "
            "relevant documents are first retrieved from a knowledge base, then passed "
            "as context to the LLM to generate a grounded, accurate answer. Reduces "
            "hallucinations and allows LLMs to access up-to-date information."
        ),
        "domain": "NLP",
    },
    {
        "id": "nlp_004",
        "name": "Tokenization",
        "type": "Concept",
        "description": (
            "Tokenization splits text into smaller units called tokens. In modern LLMs, "
            "tokens are sub-word pieces (e.g., 'running' → 'run' + 'ning'). A rough rule: "
            "1 token ≈ 4 characters or 0.75 words. Token count determines API cost and "
            "context window usage."
        ),
        "domain": "NLP",
    },
    {
        "id": "nlp_005",
        "name": "Embeddings",
        "type": "Concept",
        "description": (
            "Embeddings are dense numerical vector representations of text (or other data). "
            "Similar meanings map to nearby vectors in high-dimensional space. Used for "
            "semantic search, recommendation systems, and as input to ML models. "
            "Popular models: text-embedding-ada-002, sentence-transformers."
        ),
        "domain": "NLP",
    },

    # ── Graph Technology ─────────────────────────────────────────────────────
    {
        "id": "graph_001",
        "name": "Graph Database",
        "type": "Tool",
        "description": (
            "A graph database stores data as nodes (entities) and edges (relationships) "
            "rather than tables. Excellent for highly connected data. Supports multi-hop "
            "traversal queries that would require many JOINs in SQL. Examples: TigerGraph, "
            "Neo4j, Amazon Neptune."
        ),
        "domain": "Databases",
    },
    {
        "id": "graph_002",
        "name": "TigerGraph",
        "type": "Tool",
        "description": (
            "TigerGraph is a high-performance native parallel graph database platform. "
            "It uses GSQL (a SQL-like graph query language) and excels at real-time deep "
            "link analytics. Can perform multi-hop traversals across billions of edges "
            "in seconds. Used in fraud detection, supply chain, and AI applications."
        ),
        "domain": "Databases",
    },
    {
        "id": "graph_003",
        "name": "Knowledge Graph",
        "type": "Concept",
        "description": (
            "A Knowledge Graph is a structured representation of real-world entities and "
            "their relationships. It organizes information semantically so machines can "
            "reason about it. Used by Google Search, Siri, and enterprise AI systems to "
            "provide contextual, interconnected answers."
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
            "find multi-hop connected context. This provides richer, more structured "
            "context to LLMs, improving answer accuracy for complex questions."
        ),
        "domain": "Databases",
    },
    {
        "id": "graph_005",
        "name": "GSQL",
        "type": "Tool",
        "description": (
            "GSQL is TigerGraph's graph query language. It is Turing-complete and supports "
            "pattern matching, multi-hop traversals, aggregations, and updates on graph data. "
            "Syntax is similar to SQL but designed for graph operations. Supports parallel "
            "execution for high performance."
        ),
        "domain": "Databases",
    },

    # ── Frameworks & Tools ────────────────────────────────────────────────────
    {
        "id": "fw_001",
        "name": "TensorFlow",
        "type": "Framework",
        "description": (
            "TensorFlow is an open-source ML framework developed by Google. Supports "
            "building and training neural networks at scale. Features Keras high-level API "
            "for rapid prototyping, TFLite for mobile deployment, and TFX for production "
            "ML pipelines. Widely used in industry and research."
        ),
        "domain": "Tools",
    },
    {
        "id": "fw_002",
        "name": "PyTorch",
        "type": "Framework",
        "description": (
            "PyTorch is an open-source deep learning framework developed by Meta AI. "
            "Known for its dynamic computation graph (eager execution) making debugging "
            "intuitive. Dominant in research. Supports GPU acceleration via CUDA. "
            "Foundation of Hugging Face Transformers."
        ),
        "domain": "Tools",
    },
    {
        "id": "fw_003",
        "name": "Scikit-learn",
        "type": "Framework",
        "description": (
            "Scikit-learn is the most popular Python library for classical machine learning. "
            "Provides simple, consistent API for classification, regression, clustering, "
            "and preprocessing. Great for beginners. Includes algorithms like Random Forest, "
            "SVM, KNN, and Linear Regression. Built on NumPy and SciPy."
        ),
        "domain": "Tools",
    },
    {
        "id": "fw_004",
        "name": "Streamlit",
        "type": "Framework",
        "description": (
            "Streamlit is an open-source Python framework for building interactive web apps "
            "for data science and ML — no web development experience needed. Write pure Python "
            "and get a shareable web app. Used for dashboards, data exploration, and "
            "ML model demos."
        ),
        "domain": "Tools",
    },
    {
        "id": "fw_005",
        "name": "Groq API",
        "type": "Tool",
        "description": (
            "Groq API provides fast access to state-of-the-art AI "
            "models like LLaMA 3. Groq is optimized for extreme speed and cost efficiency. Supports "
            "various modalities through different endpoints."
        ),
        "domain": "Tools",
    },
]


# ── RELATIONSHIPS ──────────────────────────────────────────────────────────────
# Each relationship has:
#   source     : id of the source entity
#   target     : id of the target entity
#   relation   : type of relationship (verb phrase)
#   description: explanation of the connection

RELATIONSHIPS = [
    # ML core hierarchy
    {"source": "ml_001", "target": "dl_001",  "relation": "INCLUDES",        "description": "Machine Learning includes Deep Learning as a subfield"},
    {"source": "ml_001", "target": "ml_002",  "relation": "INCLUDES",        "description": "Machine Learning includes Supervised Learning"},
    {"source": "ml_001", "target": "ml_003",  "relation": "INCLUDES",        "description": "Machine Learning includes Unsupervised Learning"},
    {"source": "ml_001", "target": "ml_004",  "relation": "INCLUDES",        "description": "Machine Learning includes Reinforcement Learning"},
    {"source": "ml_001", "target": "ml_005",  "relation": "REQUIRES",        "description": "Machine Learning requires Feature Engineering"},

    # Deep Learning connections
    {"source": "dl_001", "target": "dl_002",  "relation": "USES",            "description": "Deep Learning uses Neural Networks as its core structure"},
    {"source": "dl_001", "target": "dl_003",  "relation": "INCLUDES",        "description": "Deep Learning includes CNNs for image tasks"},
    {"source": "dl_001", "target": "dl_004",  "relation": "USES",            "description": "Deep Learning uses Transformer Architecture for sequence tasks"},
    {"source": "dl_002", "target": "dl_005",  "relation": "TRAINED_WITH",    "description": "Neural Networks are trained with Backpropagation"},

    # NLP connections
    {"source": "nlp_001", "target": "dl_001", "relation": "USES",            "description": "NLP uses Deep Learning for language understanding"},
    {"source": "nlp_001", "target": "nlp_002","relation": "USES",            "description": "NLP uses Large Language Models"},
    {"source": "nlp_002", "target": "dl_004", "relation": "BUILT_ON",        "description": "LLMs are built on the Transformer Architecture"},
    {"source": "nlp_002", "target": "nlp_004","relation": "USES",            "description": "LLMs use Tokenization to process text"},
    {"source": "nlp_003", "target": "nlp_002","relation": "ENHANCES",        "description": "RAG enhances LLMs with retrieved context"},
    {"source": "nlp_003", "target": "graph_004","relation": "EXTENDED_BY",   "description": "Traditional RAG is extended by GraphRAG"},
    {"source": "nlp_005", "target": "nlp_001","relation": "ENABLES",         "description": "Embeddings enable semantic NLP tasks"},

    # Graph connections
    {"source": "graph_004", "target": "graph_001","relation": "USES",        "description": "GraphRAG uses a Graph Database for retrieval"},
    {"source": "graph_004", "target": "graph_003","relation": "USES",        "description": "GraphRAG uses Knowledge Graphs"},
    {"source": "graph_002", "target": "graph_001","relation": "IS_A",        "description": "TigerGraph is a type of Graph Database"},
    {"source": "graph_002", "target": "graph_005","relation": "USES",        "description": "TigerGraph uses GSQL as its query language"},
    {"source": "graph_003", "target": "nlp_005", "relation": "USES",         "description": "Knowledge Graphs often use Embeddings for similarity"},

    # Framework connections
    {"source": "fw_001",  "target": "ml_001",  "relation": "IMPLEMENTS",     "description": "TensorFlow implements Machine Learning algorithms"},
    {"source": "fw_002",  "target": "dl_001",  "relation": "IMPLEMENTS",     "description": "PyTorch implements Deep Learning models"},
    {"source": "fw_003",  "target": "ml_002",  "relation": "IMPLEMENTS",     "description": "Scikit-learn implements Supervised Learning algorithms"},
    {"source": "fw_003",  "target": "ml_003",  "relation": "IMPLEMENTS",     "description": "Scikit-learn implements Unsupervised Learning algorithms"},
    {"source": "fw_005",  "target": "nlp_002", "relation": "PROVIDES_ACCESS","description": "Groq API provides access to Large Language Models"},
    {"source": "fw_004",  "target": "graph_004","relation": "USED_FOR",      "description": "Streamlit is used for building GraphRAG demo dashboards"},
]


def get_all_entities():
    """Return the full list of entity dictionaries."""
    return ENTITIES


def get_all_relationships():
    """Return the full list of relationship dictionaries."""
    return RELATIONSHIPS


def get_entity_by_id(entity_id: str):
    """Return a single entity dict by its ID, or None if not found."""
    for e in ENTITIES:
        if e["id"] == entity_id:
            return e
    return None
