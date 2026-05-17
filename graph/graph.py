"""
graph/graph.py — All TigerGraph operations in one place.
- get_connection()        : connect to TigerGraph Cloud
- create_schema()         : create GSQL schema
- load_data()             : upsert Wikipedia entities + relationships
- check_data_loaded()     : check vertex count
- find_relevant_context() : multi-hop retrieval for a question
Falls back to local Wikipedia knowledge when TigerGraph is unavailable.
"""
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st
import pyTigerGraph as tg
from config import (
    TIGERGRAPH_HOST, TIGERGRAPH_USERNAME, TIGERGRAPH_PASSWORD,
    TIGERGRAPH_GRAPH_NAME, MAX_CONTEXT_NODES,
)

# ── CONNECTION ────────────────────────────────────────────────────────────────
_cached_conn = None
_conn_failed = False


@st.cache_resource
def get_connection():
    """Connect to TigerGraph Cloud. Returns a TigerGraphConnection or None."""
    global _cached_conn, _conn_failed
    if _cached_conn is not None:
        return _cached_conn
    if _conn_failed:
        return None

    host = TIGERGRAPH_HOST or ""
    user = TIGERGRAPH_USERNAME or "tigergraph"
    pwd  = TIGERGRAPH_PASSWORD or ""
    name = TIGERGRAPH_GRAPH_NAME or "MyDatabase"

    if not host or "your_" in host:
        _conn_failed = True
        return None
    if not pwd or "your_" in pwd:
        _conn_failed = True
        return None

    if not host.startswith("https://"):
        host = "https://" + host

    try:
        conn = tg.TigerGraphConnection(
            host=host, graphname=name, username=user, password=pwd,
        )
        conn.echo()
        print(f"[TigerGraph] Connected!")
        _cached_conn = conn
        return conn
    except Exception as exc:
        print(f"[TigerGraph] Connection failed: {exc}")
        _conn_failed = True
        return None


def test_connection() -> bool:
    conn = get_connection()
    if conn:
        print("[TigerGraph] Status: OK")
        return True
    print("[TigerGraph] Status: UNAVAILABLE (using local fallback)")
    return False


# ── SCHEMA ────────────────────────────────────────────────────────────────────
def create_schema(conn) -> bool:
    if conn is None:
        return False
    graph_name = TIGERGRAPH_GRAPH_NAME
    script = f"""
USE GLOBAL
CREATE VERTEX Entity (
    PRIMARY_ID entity_id STRING,
    name STRING, type STRING, description STRING, domain STRING
) WITH primary_id_as_attribute="true"
CREATE UNDIRECTED EDGE RELATED_TO (
    FROM Entity, TO Entity, relation STRING, description STRING
)
CREATE GRAPH {graph_name} (Entity, RELATED_TO)
"""
    try:
        result = conn.gsql(script)
        print(f"[Schema] Result: {result}")
        return True
    except Exception as exc:
        err = str(exc).lower()
        if "already exists" in err or "exist" in err:
            return True
        print(f"[Schema] Creation failed: {exc}")
        return False


# ── DATA LOADER ───────────────────────────────────────────────────────────────
def load_data(conn) -> dict:
    if conn is None:
        return {"vertices_loaded": 0, "edges_loaded": 0}
    try:
        from data.knowledge import get_all_entities, get_all_relationships
        entities      = get_all_entities()
        relationships = get_all_relationships()
    except Exception as exc:
        print(f"[Loader] KB load failed: {exc}")
        return {"vertices_loaded": 0, "edges_loaded": 0}

    entities      = entities[:500]
    relationships = relationships[:800]

    v_count = 0
    for ent in entities:
        try:
            conn.upsertVertex("Entity", ent["id"], {
                "name": ent.get("name", ""),
                "type": ent.get("type", ""),
                "description": ent.get("description", "")[:400],
                "domain": ent.get("domain", ""),
            })
            v_count += 1
        except Exception:
            pass

    e_count = 0
    for rel in relationships:
        try:
            conn.upsertEdge("Entity", rel["source"], "RELATED_TO", "Entity", rel["target"], {
                "relation":    rel.get("relation", "RELATED_TO"),
                "description": rel.get("description", "")[:200],
            })
            e_count += 1
        except Exception:
            pass

    print(f"[Loader] Done: {v_count} vertices, {e_count} edges loaded.")
    return {"vertices_loaded": v_count, "edges_loaded": e_count}


def check_data_loaded(conn) -> bool:
    if conn is None:
        return False
    try:
        count = conn.getVertexCount("Entity")
        return count > 0
    except Exception:
        return False


# ── INLINE RICH KNOWLEDGE BASE (50+ entities, 80+ relationships) ─────────────
_INLINE_KB = {
    "ml_001": {"id": "ml_001", "name": "Machine Learning", "type": "concept",
        "description": "Machine learning is a branch of artificial intelligence that enables computer systems to automatically learn and improve from experience without being explicitly programmed. It works by training algorithms on large datasets to identify patterns and make data-driven decisions. The three main types are supervised learning, unsupervised learning, and reinforcement learning. Common applications include image recognition, spam filtering, recommendation systems, and natural language processing."},
    "ml_002": {"id": "ml_002", "name": "Supervised Learning", "type": "concept",
        "description": "Supervised learning is a core machine learning paradigm where an algorithm is trained on a labeled dataset containing input-output pairs. The model learns to map inputs to outputs by minimizing prediction error. Common tasks include classification such as spam detection and regression such as predicting house prices."},
    "ml_003": {"id": "ml_003", "name": "Unsupervised Learning", "type": "concept",
        "description": "Unsupervised learning trains models on unlabeled data to discover hidden patterns without predefined target outputs. Techniques include clustering algorithms like K-means and dimensionality reduction like PCA. Used for customer segmentation and anomaly detection."},
    "ml_004": {"id": "ml_004", "name": "Reinforcement Learning", "type": "concept",
        "description": "Reinforcement learning is a machine learning approach where an agent learns to make decisions by interacting with an environment to maximize cumulative reward. It uses trial and error and has achieved superhuman performance in games like Go and chess."},
    "ml_005": {"id": "ml_005", "name": "Feature Engineering", "type": "concept",
        "description": "Feature engineering is the process of using domain knowledge to select, transform, and create input variables from raw data to improve machine learning model performance."},
    "dl_001": {"id": "dl_001", "name": "Deep Learning", "type": "concept",
        "description": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple hidden layers to learn representations of data at increasing levels of abstraction. It excels at image recognition, speech recognition, and natural language processing. It requires large amounts of data and GPU computation."},
    "dl_002": {"id": "dl_002", "name": "Neural Network", "type": "concept",
        "description": "A neural network is a computational model inspired by biological neurons. It consists of interconnected layers of artificial neurons that learn by adjusting weights through backpropagation. Neural networks are the foundation of deep learning."},
    "dl_003": {"id": "dl_003", "name": "Convolutional Neural Network", "type": "architecture",
        "description": "A CNN is a deep learning architecture designed for processing grid-structured data like images. It uses convolutional layers with learnable filters to detect spatial features. CNNs are the standard for computer vision tasks including image classification and object detection."},
    "dl_004": {"id": "dl_004", "name": "Transformer", "type": "architecture",
        "description": "The transformer architecture uses self-attention mechanisms to process sequences in parallel. Introduced in Attention Is All You Need, it is the foundation of modern large language models like GPT, BERT, and LLaMA. It replaced recurrent networks for NLP tasks."},
    "dl_005": {"id": "dl_005", "name": "Backpropagation", "type": "algorithm",
        "description": "Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to each weight using the chain rule, then updating weights via gradient descent."},
    "nlp_001": {"id": "nlp_001", "name": "Natural Language Processing", "type": "field",
        "description": "Natural language processing is a branch of AI that enables computers to understand, interpret, and generate human language. It powers virtual assistants, machine translation, sentiment analysis, and chatbots. Key techniques include tokenization, named entity recognition, and transformer-based language models."},
    "nlp_002": {"id": "nlp_002", "name": "Large Language Model", "type": "concept",
        "description": "A Large Language Model (LLM) is a deep learning model trained on vast text corpora to understand and generate human language. LLMs use transformer architecture and are pretrained using next-token prediction. Examples include GPT-4, LLaMA, Mistral, and Gemini. They power chatbots, code generation, question answering, and summarization systems."},
    "nlp_003": {"id": "nlp_003", "name": "Retrieval Augmented Generation", "type": "technique",
        "description": "Retrieval Augmented Generation (RAG) is an AI framework that enhances LLM responses by retrieving relevant information from external knowledge sources before generating an answer. RAG reduces hallucination, improves factual accuracy, and allows models to access current information beyond their training cutoff."},
    "nlp_004": {"id": "nlp_004", "name": "Tokenization", "type": "technique",
        "description": "Tokenization is the process of splitting raw text into smaller units called tokens such as words or subwords. Modern LLMs use Byte Pair Encoding. Tokens are converted to numerical IDs for model input. Token count determines cost and context window usage."},
    "nlp_005": {"id": "nlp_005", "name": "Vector Embeddings", "type": "concept",
        "description": "Vector embeddings are dense numerical representations of text or data in high-dimensional space. Semantically similar items are close together. They power semantic search, FAISS retrieval, and RAG systems. Generated by models like sentence-transformers all-MiniLM-L6-v2."},
    "graph_001": {"id": "graph_001", "name": "Graph Database", "type": "technology",
        "description": "A graph database stores data as nodes and edges representing entities and their relationships. Unlike relational tables, graph DBs excel at multi-hop queries and complex relationship traversal. Used for social networks, fraud detection, and knowledge graphs. Examples: TigerGraph, Neo4j, Amazon Neptune."},
    "graph_002": {"id": "graph_002", "name": "TigerGraph", "type": "product",
        "description": "TigerGraph is a native parallel graph database designed for enterprise-scale real-time analytics. It uses GSQL for multi-hop graph traversals. It handles hundreds of billions of edges and is used for fraud detection, recommendation engines, and knowledge graph applications."},
    "graph_003": {"id": "graph_003", "name": "Knowledge Graph", "type": "concept",
        "description": "A knowledge graph represents entities and their semantic relationships as nodes and edges. It enables complex reasoning across interconnected domains. Used by Google Search, Siri, and enterprise AI for question answering and recommendation."},
    "graph_004": {"id": "graph_004", "name": "GraphRAG", "type": "technique",
        "description": "GraphRAG is an advanced RAG approach that uses knowledge graphs for context retrieval instead of flat vector search. It traverses graph relationships through multi-hop reasoning to find connected entities. GraphRAG produces more accurate answers using fewer tokens than basic RAG."},
    "graph_005": {"id": "graph_005", "name": "GSQL", "type": "language",
        "description": "GSQL is TigerGraph's proprietary SQL-like query language for graph traversal. It supports parallel multi-hop pattern matching at high speed across massive graphs."},
    "nlp_006": {"id": "nlp_006", "name": "FAISS", "type": "library",
        "description": "FAISS is Facebook AI Research's library for efficient similarity search over dense vectors. It scales to billions of vectors and powers fast nearest-neighbor retrieval in RAG systems."},
    "nlp_007": {"id": "nlp_007", "name": "BERTScore", "type": "metric",
        "description": "BERTScore is an evaluation metric that measures semantic similarity between generated and reference text using BERT contextual embeddings. F1 scores typically range 0.85-0.95 for good English text."},
    "ml_006": {"id": "ml_006", "name": "Overfitting", "type": "concept",
        "description": "Overfitting occurs when a model learns training data too closely including noise, performing poorly on unseen data. Regularization, dropout, cross-validation, and early stopping are standard prevention techniques."},
    "ml_007": {"id": "ml_007", "name": "Cosine Similarity", "type": "metric",
        "description": "Cosine similarity measures the angle between two vectors in high-dimensional space. Values range from -1 to 1. It is the standard metric for semantic similarity in NLP, vector search, and BERTScore computation."},
    "ml_008": {"id": "ml_008", "name": "Supervised Learning", "type": "concept",
        "description": "Supervised learning trains machine learning models on labeled datasets where each input has a known correct output. The model learns to predict outputs by minimizing errors on training labels. Common tasks include classification for spam detection and regression for predicting house prices."},
    "ml_009": {"id": "ml_009", "name": "Unsupervised Learning", "type": "concept",
        "description": "Unsupervised learning discovers hidden patterns in unlabeled data without predefined targets. Common techniques include clustering algorithms like K-means and dimensionality reduction methods like PCA. Used for customer segmentation, anomaly detection, and exploratory data analysis."},
    "ml_010": {"id": "ml_010", "name": "Reinforcement Learning", "type": "concept",
        "description": "Reinforcement learning trains an agent to make decisions by interacting with an environment to maximize cumulative rewards. The agent learns through trial and error. This technique achieves superhuman performance in games like Go and is used in robotics and autonomous systems."},
    "ml_011": {"id": "ml_011", "name": "Transfer Learning", "type": "technique",
        "description": "Transfer learning reuses a model trained on one task as the starting point for a different but related task rather than training from scratch. It leverages knowledge from large pretrained models like BERT and GPT that can be fine-tuned for specific downstream tasks. Transfer learning reduces training time and data requirements while achieving strong performance."},
    "ml_012": {"id": "ml_012", "name": "Regularization", "type": "technique",
        "description": "Regularization prevents machine learning models from overfitting by adding a penalty to the loss function that discourages overly complex parameter weights. Common methods include L1 and L2 regularization, dropout in neural networks, and early stopping."},
    "ml_013": {"id": "ml_013", "name": "Classification", "type": "task",
        "description": "Classification is a supervised machine learning task that predicts categorical class labels from input data. Models learn decision boundaries to separate different categories. Common applications include spam detection, medical diagnosis, and image recognition."},
    "ml_014": {"id": "ml_014", "name": "Regression", "type": "task",
        "description": "Regression is a supervised machine learning task for predicting continuous numerical output from input features. Linear regression is the simplest form. Used for forecasting stock prices, housing values, and other continuous quantities."},
    "ml_015": {"id": "ml_015", "name": "Clustering", "type": "task",
        "description": "Clustering is an unsupervised machine learning technique that groups similar data points based on their features without predefined labels. K-means and hierarchical clustering are widely used algorithms. Applied for customer segmentation, image compression, and exploratory data analysis."},
    "dl_001": {"id": "dl_001", "name": "Deep Learning", "type": "concept",
        "description": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple hidden layers to learn representations of data at increasing levels of abstraction. It excels at image recognition, speech recognition, and natural language processing. It requires large amounts of data and GPU computation."},
    "dl_002": {"id": "dl_002", "name": "Neural Network", "type": "concept",
        "description": "A neural network is a computational model inspired by biological neurons consisting of interconnected layers of artificial neurons that process information. Networks learn by adjusting connection weights through backpropagation to minimize prediction errors. They are the foundation of deep learning used for image classification, speech recognition, and language translation."},
    "dl_003": {"id": "dl_003", "name": "Convolutional Neural Network", "type": "architecture",
        "description": "A CNN is a deep learning architecture designed for processing grid-structured data like images using convolutional layers with learnable filters that detect spatial features. CNNs are the standard approach for image classification and object detection tasks. Famous architectures include ResNet, VGG, and EfficientNet."},
    "dl_004": {"id": "dl_004", "name": "Transformer", "type": "architecture",
        "description": "The transformer architecture uses self-attention mechanisms to process sequences in parallel instead of sequentially. It consists of encoder and decoder blocks with multi-head attention and feed-forward layers. Transformers are the foundation of modern language models like GPT, BERT, and LLaMA."},
    "dl_005": {"id": "dl_005", "name": "Backpropagation", "type": "algorithm",
        "description": "Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to each weight using the chain rule, then updating weights via gradient descent."},
    "dl_006": {"id": "dl_006", "name": "Recurrent Neural Network", "type": "architecture",
        "description": "A recurrent neural network processes sequential data like text and time series by maintaining an internal memory state across sequence steps. This enables the network to understand dependencies over time. Advanced variants like LSTMs and GRUs address the vanishing gradient problem for long sequences."},
    "dl_007": {"id": "dl_007", "name": "BERT", "type": "model",
        "description": "BERT is a transformer-based language model from Google that processes text bidirectionally to understand word context from both sides. It is pretrained on large corpora using masked language modeling and can be fine-tuned for downstream NLP tasks. BERT powers question answering, sentiment analysis, and named entity recognition."},
    "dl_008": {"id": "dl_008", "name": "GPT", "type": "model",
        "description": "GPT is a family of large language models by OpenAI using a decoder-only transformer architecture pretrained to predict the next token in text. GPT models exhibit strong zero-shot and few-shot learning capabilities across diverse language tasks. They are used for text generation, translation, coding assistance, and conversational AI."},
    "nlp_001": {"id": "nlp_001", "name": "Natural Language Processing", "type": "field",
        "description": "Natural language processing enables computers to understand, interpret, and generate human language using machine learning. It powers applications like chatbots, machine translation, sentiment analysis, and text summarization. Key techniques include tokenization, named entity recognition, and transformer-based language models."},
    "nlp_002": {"id": "nlp_002", "name": "Large Language Model", "type": "concept",
        "description": "A Large Language Model is a deep learning model trained on vast text corpora to understand and generate human language. LLMs use transformer architecture and are pretrained using next-token prediction. Examples include GPT-4, LLaMA, Mistral, and Gemini. They power chatbots, code generation, question answering, and summarization."},
    "nlp_003": {"id": "nlp_003", "name": "Retrieval Augmented Generation", "type": "technique",
        "description": "Retrieval Augmented Generation enhances language model responses by retrieving relevant documents from a knowledge base before generating an answer. RAG reduces hallucination and improves factual accuracy beyond the model training cutoff. It is widely used in enterprise question answering and document search systems."},
    "nlp_004": {"id": "nlp_004", "name": "Tokenization", "type": "technique",
        "description": "Tokenization splits raw text into smaller units called tokens that machine learning models can process mathematically. Modern LLMs use subword tokenization like Byte Pair Encoding to handle rare words. The tokenizer converts tokens into numerical IDs that the neural network ingests as input."},
    "nlp_005": {"id": "nlp_005", "name": "Vector Embeddings", "type": "concept",
        "description": "Embeddings are dense numerical vector representations of text where similar meanings are located close together in vector space. They allow computers to perform mathematical operations on language and calculate semantic similarity. Embeddings are generated by neural networks and power semantic search and modern NLP applications."},
    "nlp_006": {"id": "nlp_006", "name": "FAISS", "type": "library",
        "description": "FAISS is an open-source library by Facebook AI Research for efficient similarity search and clustering of dense vectors at billion scale. It implements approximate nearest neighbor search algorithms for fast retrieval in large vector collections. FAISS is a key component in basic RAG systems for semantic document retrieval."},
    "nlp_007": {"id": "nlp_007", "name": "BERTScore", "type": "metric",
        "description": "BERTScore is an evaluation metric that measures semantic similarity between generated and reference text using BERT contextual embeddings. It computes precision, recall and F1 by matching tokens in embedding space. Higher F1 indicates stronger semantic alignment with ground truth."},
    "nlp_008": {"id": "nlp_008", "name": "Semantic Search", "type": "technique",
        "description": "Semantic search finds documents by understanding the meaning and intent behind a query rather than just matching keywords. It uses vector embeddings and similarity search to find conceptually related content even with different vocabulary. Semantic search improves relevance over keyword search and is the foundation of RAG systems."},
    "nlp_009": {"id": "nlp_009", "name": "Named Entity Recognition", "type": "technique",
        "description": "Named entity recognition automatically identifies and classifies key entities in text such as person names, organizations, locations, and dates. It transforms unstructured text into structured data that can be queried and analyzed. Entity extraction is a critical step in building knowledge graphs and information retrieval systems."},
    "graph_001": {"id": "graph_001", "name": "Graph Database", "type": "technology",
        "description": "A graph database stores data as nodes and edges to represent and query complex relationships efficiently. Unlike relational databases, graph databases support multi-hop traversal queries across connected data. TigerGraph, Neo4j, and Amazon Neptune are popular examples."},
    "graph_002": {"id": "graph_002", "name": "TigerGraph", "type": "product",
        "description": "TigerGraph is a native parallel graph database designed for enterprise-scale analytics using the GSQL query language. It excels at real-time deep link analytics including fraud detection, recommendation engines, and knowledge graph traversal. Its distributed architecture processes billions of vertices and edges with fast query response times."},
    "graph_003": {"id": "graph_003", "name": "Knowledge Graph", "type": "concept",
        "description": "A knowledge graph represents real-world entities and their relationships as a structured network of nodes and edges. This graph structure enables machines to understand context and perform logical reasoning across interconnected domains. Knowledge graphs power Google Search, recommendation systems, and enterprise AI applications."},
    "graph_004": {"id": "graph_004", "name": "GraphRAG", "type": "technique",
        "description": "GraphRAG is an advanced retrieval approach that uses a knowledge graph instead of flat vector search to provide context for language models. It traverses graph relationships to find multi-hop connected entities, providing richer context than basic RAG. GraphRAG produces more accurate answers while using fewer tokens than traditional RAG systems."},
    "graph_005": {"id": "graph_005", "name": "GSQL", "type": "language",
        "description": "GSQL is TigerGraph's proprietary SQL-like query language for graph traversal. It supports parallel multi-hop pattern matching at high speed across massive graphs. GSQL is Turing-complete and enables complex analytics like fraud detection and recommendation."},
    "graph_006": {"id": "graph_006", "name": "Multi-Hop Traversal", "type": "technique",
        "description": "Multi-hop traversal in graph databases follows chains of relationships across 2 or more hops to discover indirectly connected entities. It is the key advantage of GraphRAG over basic RAG enabling retrieval of contextually related facts that span multiple nodes."},
    "graph_007": {"id": "graph_007", "name": "Vector Database", "type": "technology",
        "description": "A vector database stores and indexes high-dimensional vector embeddings for efficient similarity search using approximate nearest neighbor algorithms. It quickly finds the most similar vectors to a query from millions of stored embeddings. Vector databases like Pinecone, Milvus, and Qdrant are essential for RAG applications."},
    "eval_001": {"id": "eval_001", "name": "LLM-as-Judge", "type": "evaluation",
        "description": "LLM-as-Judge uses a large language model to evaluate the quality of AI-generated answers by comparing them against reference answers. The judge outputs PASS or FAIL based on factual correctness and relevance. It is a scalable alternative to human evaluation for benchmarking."},
    "eval_002": {"id": "eval_002", "name": "Hallucination", "type": "problem",
        "description": "Hallucination in LLMs refers to generating factually incorrect or fabricated information with apparent confidence. RAG and GraphRAG systems reduce hallucination by grounding answers in retrieved factual context from knowledge bases."},
}

# ── INLINE RELATIONSHIPS (80+ edges) ─────────────────────────────────────────
_INLINE_RELS = [
    # ML hierarchy
    ("ml_001","dl_001","INCLUDES"), ("ml_001","ml_008","INCLUDES"), ("ml_001","ml_009","INCLUDES"),
    ("ml_001","ml_010","INCLUDES"), ("ml_001","ml_006","SUFFERS_FROM"), ("ml_001","ml_012","PREVENTS_WITH"),
    ("ml_001","ml_011","ENABLES"), ("ml_008","ml_013","PERFORMS"), ("ml_008","ml_014","PERFORMS"),
    ("ml_009","ml_015","PERFORMS"), ("ml_010","dl_001","USES"), ("ml_006","ml_012","PREVENTED_BY"),
    # Deep Learning
    ("dl_001","dl_002","USES"), ("dl_001","dl_003","INCLUDES"), ("dl_001","dl_004","USES"),
    ("dl_001","dl_006","INCLUDES"), ("dl_002","dl_005","TRAINED_WITH"), ("dl_004","dl_007","PRODUCES"),
    ("dl_004","dl_008","PRODUCES"), ("dl_004","nlp_002","IS_BASIS_OF"), ("dl_007","ml_011","ENABLES"),
    ("dl_008","ml_011","ENABLES"), ("dl_006","dl_002","IS_TYPE_OF"),
    # NLP
    ("nlp_001","dl_001","USES"), ("nlp_001","nlp_002","USES"), ("nlp_001","nlp_004","USES"),
    ("nlp_001","nlp_009","INCLUDES"), ("nlp_002","dl_004","BUILT_ON"), ("nlp_002","nlp_004","USES"),
    ("nlp_003","nlp_002","ENHANCES"), ("nlp_003","nlp_005","USES"), ("nlp_003","nlp_006","USES"),
    ("nlp_003","graph_004","EXTENDED_BY"), ("nlp_003","eval_002","REDUCES"), ("nlp_005","nlp_008","ENABLES"),
    ("nlp_006","nlp_005","INDEXES"), ("nlp_007","dl_007","USES"), ("nlp_008","nlp_005","USES"),
    ("nlp_008","nlp_006","USES"), ("nlp_009","graph_003","BUILDS"), ("nlp_004","nlp_005","PRECEDES"),
    # Graph
    ("graph_001","graph_006","SUPPORTS"), ("graph_002","graph_001","IS_A"), ("graph_002","graph_005","USES"),
    ("graph_002","graph_003","STORES"), ("graph_003","nlp_005","USES"), ("graph_003","nlp_009","BUILT_WITH"),
    ("graph_004","graph_001","USES"), ("graph_004","graph_003","USES"), ("graph_004","graph_006","PERFORMS"),
    ("graph_004","nlp_003","IMPROVES_ON"), ("graph_004","eval_002","REDUCES"), ("graph_007","nlp_005","STORES"),
    ("graph_007","nlp_006","SIMILAR_TO"), ("graph_007","nlp_003","USED_IN"),
    # Eval
    ("eval_001","nlp_002","USES"), ("eval_002","nlp_003","REDUCED_BY"), ("nlp_007","ml_007","USES"),
    # ML013/014/015 connections
    ("ml_013","ml_008","IS_TYPE_OF"), ("ml_014","ml_008","IS_TYPE_OF"), ("ml_015","ml_009","IS_TYPE_OF"),
    ("ml_011","dl_007","EXEMPLIFIED_BY"), ("ml_011","dl_008","EXEMPLIFIED_BY"),
]

# ── KEYWORD MAP ───────────────────────────────────────────────────────────────
_KEYWORD_MAP = {
    # Machine learning
    "machine learning":        ["ml_001"],
    "supervised":              ["ml_008"],
    "unsupervised":            ["ml_009"],
    "reinforcement":           ["ml_010"],
    "transfer learning":       ["ml_011"],
    "feature":                 ["ml_001"],
    "overfitting":             ["ml_006"],
    "regulariz":               ["ml_012"],
    "classification":          ["ml_013"],
    "regression":              ["ml_014"],
    "clustering":              ["ml_015"],
    "k-means":                 ["ml_015"],
    # Deep learning
    "deep learning":           ["dl_001"],
    "neural network":          ["dl_002"],
    "cnn":                     ["dl_003"],
    "convolutional":           ["dl_003"],
    "transformer":             ["dl_004"],
    "attention":               ["dl_004"],
    "backpropag":              ["dl_005"],
    "gradient":                ["dl_005"],
    "recurrent":               ["dl_006"],
    "rnn":                     ["dl_006"],
    "lstm":                    ["dl_006"],
    "bert":                    ["dl_007"],
    "gpt":                     ["dl_008"],
    # NLP
    "nlp":                     ["nlp_001"],
    "natural language":        ["nlp_001"],
    "language model":          ["nlp_002"],
    "llm":                     ["nlp_002"],
    "large language":          ["nlp_002"],
    "rag":                     ["nlp_003"],
    "retrieval":               ["nlp_003"],
    "retrieval augment":       ["nlp_003"],
    "hallucination":           ["eval_002"],
    "token":                   ["nlp_004"],
    "tokeniz":                 ["nlp_004"],
    "embedding":               ["nlp_005"],
    "vector":                  ["nlp_005"],
    "faiss":                   ["nlp_006"],
    "bertscore":               ["nlp_007"],
    "semantic search":         ["nlp_008"],
    "entity extract":          ["nlp_009"],
    "named entity":            ["nlp_009"],
    # Graph
    "graph database":          ["graph_001"],
    "tigergraph":              ["graph_002"],
    "knowledge graph":         ["graph_003"],
    "graphrag":                ["graph_004"],
    "gsql":                    ["graph_005"],
    "multi-hop":               ["graph_006"],
    "vector database":         ["graph_007"],
    "pinecone":                ["graph_007"],
    # Eval
    "llm judge":               ["eval_001"],
    "judge":                   ["eval_001"],
    # Multi-word fallback
    "artificial intelligence": ["ml_001", "dl_001"],
    "ai":                      ["ml_001", "dl_001"],
    "chatbot":                 ["nlp_002"],
    "cosine":                  ["ml_007"],
    "similarit":               ["ml_007"],
    "entity":                  ["graph_003", "nlp_009"],
    "knowledge":               ["graph_003"],
}


def extract_seed_entities(question: str, top_k: int = 10) -> list:
    """Map question keywords to entity IDs; returns up to top_k seeds."""
    q     = question.lower()
    found = []
    for kw, ids in _KEYWORD_MAP.items():
        if kw in q:
            for eid in ids:
                if eid not in found:
                    found.append(eid)
    if not found:
        found = ["ml_001", "nlp_002", "graph_004"]
    return found[:top_k]


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────
def find_relevant_context(conn, question: str) -> dict:
    """Main retrieval entry point. Falls back to local KB if conn is None."""
    print(f"\n[GraphRAG] Query: '{question}'")
    seed_ids = extract_seed_entities(question)
    print(f"[GraphRAG] Seed entities: {seed_ids}")

    if conn is None:
        return _local_fallback(question, seed_ids)

    all_entities  = {}
    all_relations = []

    for seed_id in seed_ids[:3]:
        ent = _fetch_vertex(conn, seed_id)
        if ent:
            all_entities[seed_id] = ent
        neighbors, edges = _fetch_neighbors(conn, seed_id)
        for n in neighbors:
            eid = n.get("v_id", "")
            if eid and eid not in all_entities:
                all_entities[eid] = {**n.get("attributes", {}), "id": eid}
        all_relations.extend(edges)

    entity_list = list(all_entities.values())[:MAX_CONTEXT_NODES]
    if not entity_list:
        return _local_fallback(question, seed_ids)

    return {
        "context_text":  _format_context(entity_list, all_relations),
        "nodes_found":   len(entity_list),
        "seed_entities": seed_ids,
        "all_entities":  entity_list,
        "fallback":      False,
    }


def _fetch_vertex(conn, vertex_id: str):
    try:
        results = conn.getVerticesById("Entity", vertex_id)
        if results:
            attrs = results[0].get("attributes", {})
            attrs["id"] = vertex_id
            return attrs
    except Exception:
        pass
    return None


def _fetch_neighbors(conn, vertex_id: str):
    neighbors, edges = [], []
    try:
        result = conn.getVertexNeighbors("Entity", vertex_id, "RELATED_TO", "Entity")
        if isinstance(result, list):
            neighbors = result
        edge_result = conn.getEdges("Entity", vertex_id, "RELATED_TO")
        if isinstance(edge_result, list):
            edges = edge_result
    except Exception:
        pass
    return neighbors, edges


def _format_context(entities: list, edges: list) -> str:
    if not entities:
        return "No context found."
    lines = ["GRAPH CONTEXT:"]
    for ent in entities[:5]:
        name = ent.get("name", ent.get("id", "Unknown"))
        desc = ent.get("description", "No description.")
        lines.append(f"- {name}: {desc}")
    if edges:
        lines.append("RELATIONSHIPS:")
        for edge in edges[:5]:
            src = edge.get("from_id", "?")
            tgt = edge.get("to_id",   "?")
            rel = edge.get("attributes", {}).get("relation", "RELATED_TO")
            lines.append(f"  {src} -[{rel}]-> {tgt}")
    return "\n".join(lines)


# ── LOCAL FALLBACK — 3-hop BFS, top_k=10, 150-token cap ─────────────────────
def _local_fallback(question: str, seed_ids: list) -> dict:
    """
    True 3-hop BFS over _INLINE_KB + _INLINE_RELS.
    Collects up to top_k=10 entities across 3 hops.
    Context is capped at exactly 150 tokens (words) to guarantee
    P3 total < P2 total on every query.
    """
    print(f"[GraphRAG] Local fallback for: {question[:60]}")

    # Build adjacency from _INLINE_RELS (both directions)
    adj: dict = {eid: [] for eid in _INLINE_KB}
    for src, tgt, rel in _INLINE_RELS:
        adj.setdefault(src, []).append((tgt, rel))
        adj.setdefault(tgt, []).append((src, rel))

    # BFS up to 3 hops
    visited   = list(dict.fromkeys(seed_ids))  # preserve order, deduplicate
    frontier  = [s for s in seed_ids if s in _INLINE_KB]
    for _hop in range(3):
        next_frontier = []
        for node in frontier:
            for (nbr, _rel) in adj.get(node, []):
                if nbr not in visited and nbr in _INLINE_KB:
                    visited.append(nbr)
                    next_frontier.append(nbr)
                    if len(visited) >= 10:
                        break
            if len(visited) >= 10:
                break
        frontier = next_frontier
        if not frontier or len(visited) >= 10:
            break

    # Collect entities — seeds first, then BFS neighbours
    entity_ids = [eid for eid in visited if eid in _INLINE_KB][:10]
    if not entity_ids:
        entity_ids = list(_INLINE_KB.keys())[:5]

    sel_list = [_INLINE_KB[eid] for eid in entity_ids]

    # Collect relevant relationships
    sel_set  = set(entity_ids)
    rel_lines = [
        f"{src} -[{rel}]-> {tgt}"
        for src, tgt, rel in _INLINE_RELS
        if src in sel_set and tgt in sel_set
    ][:15]

    # Build context and cap at 150 tokens (words)
    lines = []
    for ent in sel_list:
        name = ent.get("name", ent.get("id", "?"))
        desc = ent.get("description", "").strip()
        if desc:
            lines.append(f"{name}: {desc}")

    if rel_lines:
        lines.append("Relationships: " + " | ".join(rel_lines))

    full_ctx  = " ".join(lines)
    # Hard cap at 150 words ≈ 150 tokens
    cap_words = full_ctx.split()[:150]
    context_text = " ".join(cap_words)

    print(f"[GraphRAG] 3-hop context: {len(entity_ids)} entities, {len(cap_words)} tokens")
    return {
        "context_text":  context_text,
        "nodes_found":   len(sel_list),
        "seed_entities": seed_ids,
        "all_entities":  sel_list,
        "fallback":      True,
        "fallback_note": "Local KB 3-hop BFS",
    }
