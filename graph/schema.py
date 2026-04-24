"""
graph/schema.py - TigerGraph Schema Definition
===============================================
Creates the vertex (node) and edge types in TigerGraph.
The schema defines the structure of the graph — like
CREATE TABLE in SQL, but for a graph database.

Beginner tip:
  - Vertex = a node (an entity like "Machine Learning")
  - Edge   = a connection between two nodes
  - Attribute = a property of a vertex or edge (like a column)
"""

import sys
import pyTigerGraph as tg


# ── SCHEMA DEFINITION ─────────────────────────────────────────────────────────

VERTEX_TYPES = [
    {
        "name": "Entity",
        "primary_id": "entity_id",
        "primary_id_as_attribute": True,
        "attributes": {
            "name":        "STRING",
            "type":        "STRING",
            "description": "STRING",
            "domain":      "STRING",
        },
    }
]

EDGE_TYPES = [
    {
        "name":      "RELATED_TO",
        "directed":  False,          # Undirected: relationship goes both ways
        "from_vertex": "Entity",
        "to_vertex":   "Entity",
        "attributes": {
            "relation":    "STRING",
            "description": "STRING",
        },
    }
]


def create_schema(conn: tg.TigerGraphConnection) -> bool:
    """
    Create the graph schema in TigerGraph.
    This includes vertex types and edge types.

    Args:
        conn: An active TigerGraphConnection object.

    Returns:
        True if schema creation succeeded, False otherwise.
    """
    print("\n📐 Creating TigerGraph schema ...")

    try:
        # Build the GSQL DDL (Data Definition Language) string
        # GSQL is TigerGraph's query language — similar to SQL
        gsql_script = _build_schema_gsql()

        print("   Running GSQL schema creation script ...")
        result = conn.gsql(gsql_script)
        print(f"   GSQL result: {result}")

        print("✅ Schema created successfully.")
        return True

    except Exception as e:
        err_str = str(e)
        # "already exists" errors are safe to ignore
        if "already exists" in err_str.lower() or "exist" in err_str.lower():
            print("ℹ️  Schema already exists — skipping creation.")
            return True
        print(f"❌ Schema creation failed: {e}")
        return False


def _build_schema_gsql() -> str:
    """
    Builds the GSQL script string for schema creation.
    Returns the multi-line GSQL DDL string.
    """
    graph_name = "GraphRAGDemo"

    script = f"""
USE GLOBAL

CREATE VERTEX Entity (
    PRIMARY_ID entity_id STRING,
    name        STRING,
    type        STRING,
    description STRING,
    domain      STRING
) WITH primary_id_as_attribute="true"

CREATE UNDIRECTED EDGE RELATED_TO (
    FROM Entity, TO Entity,
    relation    STRING,
    description STRING
)

CREATE GRAPH {graph_name} (Entity, RELATED_TO)
"""
    return script


def drop_schema(conn: tg.TigerGraphConnection) -> bool:
    """
    Drop the graph and all its data. Use with caution!
    Useful for resetting during development.
    """
    print("🗑️  Dropping existing schema ...")
    try:
        graph_name = conn.graphname
        conn.gsql(f"DROP GRAPH {graph_name}")
        conn.gsql("DROP VERTEX Entity")
        conn.gsql("DROP EDGE RELATED_TO")
        print("✅ Schema dropped.")
        return True
    except Exception as e:
        print(f"⚠️  Drop schema error (may be safe to ignore): {e}")
        return False
