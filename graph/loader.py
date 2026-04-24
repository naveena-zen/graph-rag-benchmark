"""
graph/loader.py - Data Loader for TigerGraph
=============================================
Loads the knowledge base (from data/knowledge.py) into
TigerGraph as vertices (nodes) and edges (relationships).

Beginner tip: Think of this like INSERT statements in SQL —
we're inserting our knowledge data into the graph database.
pyTigerGraph provides upsertVertex() and upsertEdge() methods.
"""

import pyTigerGraph as tg
from data.knowledge import get_all_entities, get_all_relationships


def load_data(conn: tg.TigerGraphConnection) -> dict:
    """
    Load all entities and relationships into TigerGraph.

    Args:
        conn: An active TigerGraphConnection.

    Returns:
        dict with 'vertices_loaded' and 'edges_loaded' counts.
    """
    print("\n📥 Loading knowledge data into TigerGraph ...")

    entities      = get_all_entities()
    relationships = get_all_relationships()

    vertices_loaded = _load_entities(conn, entities)
    edges_loaded    = _load_relationships(conn, relationships)

    result = {
        "vertices_loaded": vertices_loaded,
        "edges_loaded":    edges_loaded,
    }
    print(f"✅ Load complete: {vertices_loaded} vertices, {edges_loaded} edges.")
    return result


def _load_entities(conn: tg.TigerGraphConnection, entities: list) -> int:
    """
    Insert entity nodes into TigerGraph as 'Entity' vertices.
    Uses upsertVertex — inserts if new, updates if already exists.
    """
    count = 0
    print(f"   Loading {len(entities)} entity vertices ...")

    for entity in entities:
        try:
            conn.upsertVertex(
                vertexType="Entity",
                vertexId=entity["id"],
                attributes={
                    "name":        entity["name"],
                    "type":        entity["type"],
                    "description": entity["description"],
                    "domain":      entity["domain"],
                },
            )
            count += 1
            print(f"   ✔ Vertex: {entity['name']} ({entity['id']})")

        except Exception as e:
            print(f"   ⚠️  Failed to load entity '{entity['id']}': {e}")

    return count


def _load_relationships(conn: tg.TigerGraphConnection, relationships: list) -> int:
    """
    Insert relationship edges into TigerGraph as 'RELATED_TO' edges.
    Uses upsertEdge — inserts if new, updates if already exists.
    """
    count = 0
    print(f"   Loading {len(relationships)} relationship edges ...")

    for rel in relationships:
        try:
            conn.upsertEdge(
                sourceVertexType="Entity",
                sourceVertexId=rel["source"],
                edgeType="RELATED_TO",
                targetVertexType="Entity",
                targetVertexId=rel["target"],
                attributes={
                    "relation":    rel["relation"],
                    "description": rel["description"],
                },
            )
            count += 1
            print(f"   ✔ Edge: {rel['source']} --[{rel['relation']}]--> {rel['target']}")

        except Exception as e:
            print(f"   ⚠️  Failed to load edge '{rel['source']}→{rel['target']}': {e}")

    return count


def check_data_loaded(conn: tg.TigerGraphConnection) -> bool:
    """
    Quick check: does the graph already have vertices loaded?
    Returns True if data exists, False if graph is empty.
    """
    try:
        count = conn.getVertexCount("Entity")
        print(f"ℹ️  Graph currently has {count} Entity vertices.")
        return count > 0
    except Exception as e:
        print(f"⚠️  Could not check vertex count: {e}")
        return False
