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

import pyTigerGraph as tg
from config import (
    TIGERGRAPH_HOST, TIGERGRAPH_USERNAME, TIGERGRAPH_PASSWORD,
    TIGERGRAPH_GRAPH_NAME, MAX_CONTEXT_NODES,
)

# ── CONNECTION ────────────────────────────────────────────────────────────────
_cached_conn = None
_conn_failed = False


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


# ── INLINE RICH KNOWLEDGE BASE ───────────────────────────────────────────────
# FIX: This inline KB guarantees rich context is always returned even when
# data/knowledge.py entities have empty descriptions.
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
        "description": "Overfitting occurs when a model learns training data too well including noise, performing poorly on new data. It is prevented by regularization, dropout, cross-validation, and early stopping."},
    "ml_007": {"id": "ml_007", "name": "Cosine Similarity", "type": "metric",
        "description": "Cosine similarity measures the angle between two vectors in high-dimensional space. Values range from -1 to 1. It is the standard metric for semantic similarity in NLP and vector search."},
}

# ── KEYWORD MAP ───────────────────────────────────────────────────────────────
_KEYWORD_MAP = {
    "machine learning":        ["ml_001"],
    "supervised":              ["ml_002"],
    "unsupervised":            ["ml_003"],
    "reinforcement":           ["ml_004"],
    "feature":                 ["ml_005"],
    "deep learning":           ["dl_001"],
    "neural network":          ["dl_002"],
    "cnn":                     ["dl_003"],
    "convolutional":           ["dl_003"],
    "transformer":             ["dl_004"],
    "attention":               ["dl_004"],
    "backpropag":              ["dl_005"],
    "gradient":                ["dl_005"],
    "nlp":                     ["nlp_001"],
    "natural language":        ["nlp_001"],
    "language model":          ["nlp_002"],
    "llm":                     ["nlp_002"],
    "large language":          ["nlp_002"],
    "rag":                     ["nlp_003"],
    "retrieval":               ["nlp_003"],
    "retrieval augment":       ["nlp_003"],
    "token":                   ["nlp_004"],
    "tokeniz":                 ["nlp_004"],
    "embedding":               ["nlp_005"],
    "vector":                  ["nlp_005"],
    "graph database":          ["graph_001"],
    "tigergraph":              ["graph_002"],
    "knowledge graph":         ["graph_003"],
    "graphrag":                ["graph_004"],
    "gsql":                    ["graph_005"],
    "artificial intelligence": ["ml_001", "dl_001"],
    "ai":                      ["ml_001", "dl_001"],
    "classification":          ["ml_002"],
    "regression":              ["ml_002"],
    "clustering":              ["ml_003"],
    "chatbot":                 ["nlp_002"],
    "hallucination":           ["nlp_003"],
    "faiss":                   ["nlp_006"],
    "bert":                    ["dl_004", "nlp_002"],
    "gpt":                     ["nlp_002"],
    "rnn":                     ["dl_002"],
    "recurrent":               ["dl_002"],
    "overfitting":             ["ml_006"],
    "regulariz":               ["ml_006"],
    "cosine":                  ["ml_007"],
    "similarit":               ["ml_007"],
    "semantic search":         ["nlp_005", "nlp_003"],
    "entity":                  ["graph_003"],
    "knowledge":               ["graph_003"],
    "bertscore":               ["nlp_007"],
}


def extract_seed_entities(question: str) -> list:
    """Map question keywords to entity IDs."""
    q     = question.lower()
    found = []
    for kw, ids in _KEYWORD_MAP.items():
        if kw in q:
            for eid in ids:
                if eid not in found:
                    found.append(eid)
    return found if found else ["ml_001", "nlp_002", "graph_004"]


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


# ── LOCAL FALLBACK ────────────────────────────────────────────────────────────
def _local_fallback(question: str, seed_ids: list) -> dict:
    """
    FIX: Always returns rich context by combining:
      1. Inline _INLINE_KB (always available, rich descriptions)
      2. data/knowledge.py entities (if available)
    Old code returned empty strings when data/knowledge.py had thin descriptions.
    """
    print(f"[GraphRAG] Local fallback for: {question[:60]}")

    # --- Build entity map: inline KB first, then override with data/knowledge.py ---
    ent_map = dict(_INLINE_KB)  # start with inline guaranteed descriptions

    try:
        from data.knowledge import get_all_entities, get_all_relationships
        all_ents = get_all_entities()
        all_rels = get_all_relationships()
        for e in all_ents:
            eid  = e.get("id", "")
            desc = e.get("description", "").strip()
            if eid and desc and len(desc) > 30:  # only override if KB has real content
                ent_map[eid] = e
    except Exception as exc:
        print(f"[GraphRAG] data/knowledge load failed ({exc}), using inline KB only")
        all_rels = []

    # --- Select seed nodes + 1-hop neighbours ---
    selected  = list(dict.fromkeys(seed_ids))
    neighbors = set(selected)
    for rel in all_rels:
        if rel.get("source") in selected:
            neighbors.add(rel["target"])
        if rel.get("target") in selected:
            neighbors.add(rel["source"])

    # Prefer seeded nodes that exist in our map, fall back to any nodes
    final_nodes = [n for n in selected if n in ent_map]
    for n in neighbors:
        if n in ent_map and n not in final_nodes:
            final_nodes.append(n)
    if not final_nodes:
        final_nodes = list(ent_map.keys())[:5]
    final_nodes = final_nodes[:5]

    sel_list = [ent_map[eid] for eid in final_nodes if eid in ent_map]
    sel_rels  = [r for r in all_rels
                 if r.get("source") in final_nodes or r.get("target") in final_nodes]

    # --- Format rich context block ---
    lines = ["Knowledge Graph Entities:"]
    for ent in sel_list:
        name = ent.get("name", ent.get("id", "?"))
        desc = ent.get("description", "").strip()
        if desc:
            lines.append(f"\n{name}:\n{desc}")
    if sel_rels:
        lines.append("\nRelationships:")
        for rel in sel_rels[:5]:
            lines.append(f"  {rel.get('source','?')} -> [{rel.get('relation','related')}] -> {rel.get('target','?')}")

    context_text = "\n".join(lines)
    print(f"[GraphRAG] Fallback context: {len(sel_list)} entities, {len(context_text)} chars")
    return {
        "context_text":  context_text,
        "nodes_found":   len(sel_list),
        "seed_entities": seed_ids,
        "all_entities":  sel_list,
        "fallback":      True,
        "fallback_note": "Local KB (inline)",
    }
