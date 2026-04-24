"""
graph/query.py - TigerGraph Multi-Hop Query Engine
====================================================
Retrieves relevant context from TigerGraph for a user question.
Uses keyword matching + multi-hop graph traversal to find
related entities and return them as formatted text context.

Beginner tip: "Multi-hop traversal" means:
  Start at entity A → follow edges → reach B → follow more edges → reach C
  This lets us discover hidden connections that flat search would miss.
"""

import re
import pyTigerGraph as tg
import config


# Keywords that map to entity IDs in our knowledge graph
# This acts as a simple "semantic router" to find relevant starting nodes
KEYWORD_MAP = {
    # Machine Learning
    "machine learning": ["ml_001"],
    "supervised":       ["ml_002"],
    "unsupervised":     ["ml_003"],
    "reinforcement":    ["ml_004"],
    "feature":          ["ml_005"],

    # Deep Learning
    "deep learning":    ["dl_001"],
    "neural network":   ["dl_002"],
    "cnn":              ["dl_003"],
    "convolutional":    ["dl_003"],
    "transformer":      ["dl_004"],
    "attention":        ["dl_004"],
    "backpropagation":  ["dl_005"],
    "gradient":         ["dl_005"],

    # NLP
    "nlp":              ["nlp_001"],
    "natural language": ["nlp_001"],
    "language model":   ["nlp_002"],
    "llm":              ["nlp_002"],
    "large language":   ["nlp_002"],
    "rag":              ["nlp_003"],
    "retrieval":        ["nlp_003"],
    "token":            ["nlp_004"],
    "tokeniz":          ["nlp_004"],
    "embedding":        ["nlp_005"],
    "vector":           ["nlp_005"],

    # Graph
    "graph database":   ["graph_001"],
    "tigergraph":       ["graph_002"],
    "knowledge graph":  ["graph_003"],
    "graphrag":         ["graph_004"],
    "gsql":             ["graph_005"],

    # Frameworks
    "tensorflow":       ["fw_001"],
    "pytorch":          ["fw_002"],
    "scikit":           ["fw_003"],
    "sklearn":          ["fw_003"],
    "streamlit":        ["fw_004"],
    "groq":             ["fw_005"],

    # General AI
    "artificial intelligence": ["ml_001", "dl_001"],
    "ai":                      ["ml_001", "dl_001"],
    "classification":          ["ml_002"],
    "regression":              ["ml_002"],
    "clustering":              ["ml_003"],
    "image":                   ["dl_003"],
    "text":                    ["nlp_001"],
    "chatbot":                 ["nlp_002"],
    "hallucination":           ["nlp_003"],
    "context":                 ["nlp_003"],
}


def find_relevant_context(conn: tg.TigerGraphConnection, question: str) -> dict:
    """
    Main entry point: given a user question, return relevant graph context.

    Steps:
      1. Parse the question for matching keywords → find seed entity IDs
      2. Fetch those entities from TigerGraph
      3. Traverse neighbors (1-2 hops) to expand context
      4. Format everything into a readable string for the LLM

    Args:
        conn    : Active TigerGraphConnection
        question: The user's question string

    Returns:
        dict with:
            context_text   (str)  : Formatted context for the LLM prompt
            nodes_found    (int)  : Number of entities retrieved
            seed_entities  (list) : Entity IDs used as starting points
            all_entities   (list) : All entity dicts retrieved
    """
    print(f"\n🔍 GraphRAG Query: '{question}'")

    # Step 1: Find seed entity IDs from question keywords
    seed_ids = _extract_seed_entities(question)
    print(f"   Seed entities: {seed_ids}")

    if not seed_ids:
        # No keyword match → fall back to fetching some general entities
        print("   No keyword match found — using fallback general context.")
        seed_ids = _get_fallback_seeds(question)

    # Step 2: Fetch seed entities and their neighbors from TigerGraph
    all_entities  = {}
    all_relations = []

    for seed_id in seed_ids[:3]:   # Limit to 3 seeds to keep context manageable
        entity = _fetch_vertex(conn, seed_id)
        if entity:
            all_entities[seed_id] = entity

        # Fetch 1-hop neighbors
        neighbors, edges = _fetch_neighbors(conn, seed_id, hops=config.MAX_HOPS)
        for n in neighbors:
            eid = n.get("v_id", "")
            if eid and eid not in all_entities:
                all_entities[eid] = n.get("attributes", {})
                all_entities[eid]["id"] = eid
        all_relations.extend(edges)

    # Step 3: Limit total nodes
    entity_list = list(all_entities.values())[:config.MAX_CONTEXT_NODES]
    print(f"   Retrieved {len(entity_list)} entities from graph.")

    # Step 4: Format into context text
    context_text = _format_context(entity_list, all_relations)

    return {
        "context_text":  context_text,
        "nodes_found":   len(entity_list),
        "seed_entities": seed_ids,
        "all_entities":  entity_list,
    }


def _extract_seed_entities(question: str) -> list:
    """
    Scan the question for keywords from KEYWORD_MAP.
    Returns a de-duplicated list of matching entity IDs.
    """
    q_lower = question.lower()
    found   = []

    for keyword, entity_ids in KEYWORD_MAP.items():
        if keyword in q_lower:
            for eid in entity_ids:
                if eid not in found:
                    found.append(eid)

    return found


def _get_fallback_seeds(question: str) -> list:
    """
    If no keyword matched, return the most general/broad entities
    as a fallback so we always return SOME context.
    """
    return ["ml_001", "nlp_002", "graph_004"]


def _fetch_vertex(conn: tg.TigerGraphConnection, vertex_id: str) -> dict:
    """
    Fetch a single vertex from TigerGraph by its ID.
    Returns its attributes as a dict, or None if not found.
    """
    try:
        results = conn.getVerticesById("Entity", vertex_id)
        if results:
            attrs = results[0].get("attributes", {})
            attrs["id"] = vertex_id
            return attrs
    except Exception as e:
        print(f"   ⚠️  Could not fetch vertex {vertex_id}: {e}")
    return None


def _fetch_neighbors(
    conn: tg.TigerGraphConnection,
    vertex_id: str,
    hops: int = 2,
) -> tuple:
    """
    Fetch neighbor vertices connected to vertex_id via RELATED_TO edges.
    Returns (list of neighbor vertex dicts, list of edge dicts).
    """
    neighbors = []
    edges     = []

    try:
        # getVertexNeighbors returns adjacent vertices
        result = conn.getVertexNeighbors(
            vertexType="Entity",
            vertexId=vertex_id,
            edgeType="RELATED_TO",
            targetVertexType="Entity",
        )
        if isinstance(result, list):
            neighbors = result

        # Also try to get edge data
        edge_result = conn.getEdges(
            sourceVertexType="Entity",
            sourceVertexId=vertex_id,
            edgeType="RELATED_TO",
        )
        if isinstance(edge_result, list):
            edges = edge_result

    except Exception as e:
        print(f"   ⚠️  Could not fetch neighbors for {vertex_id}: {e}")

    return neighbors, edges


def _format_context(entities: list, edges: list) -> str:
    """
    Format entities and edges into a readable string for the LLM prompt.
    """
    if not entities:
        return "No relevant context found in the knowledge graph."

    lines = ["ENTITIES FROM KNOWLEDGE GRAPH:\n"]

    for i, ent in enumerate(entities, 1):
        name  = ent.get("name",        ent.get("id", "Unknown"))
        etype = ent.get("type",        "")
        desc  = ent.get("description", "No description available.")
        dom   = ent.get("domain",      "")

        lines.append(f"{i}. [{etype}] {name} (Domain: {dom})")
        lines.append(f"   {desc}\n")

    if edges:
        lines.append("\nRELATIONSHIPS BETWEEN ENTITIES:\n")
        seen = set()
        for edge in edges[:15]:   # Limit edges shown
            src  = edge.get("from_id",    "?")
            tgt  = edge.get("to_id",      "?")
            rel  = edge.get("attributes", {}).get("relation", "RELATED_TO")
            desc = edge.get("attributes", {}).get("description", "")
            key  = f"{src}-{tgt}"
            if key not in seen:
                lines.append(f"• {src} --[{rel}]--> {tgt}: {desc}")
                seen.add(key)

    return "\n".join(lines)
