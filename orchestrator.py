"""
inference/orchestrator.py - Pipeline Orchestrator
==================================================
This is the brain of the system. It coordinates both pipelines:

  Pipeline 1 (Baseline): question → Groq → answer
  Pipeline 2 (GraphRAG): question → TigerGraph → context → Groq → answer

Both run and return results for dashboard comparison.
"""

import time
from graph.connection import get_connection
from graph.query      import find_relevant_context
from llm.caller       import call_llm_baseline, call_llm_with_graph_context
from eval.metrics     import compute_metrics
import config


def run_both_pipelines(question: str) -> dict:
    """
    Run both pipelines in sequence and return combined results.
    """
    print(f"\n{'='*60}")
    print(f"🚀 Running both pipelines for: '{question}'")
    print(f"{'='*60}")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # ── Pipeline 1: Baseline LLM ─────────────────────────────────────────────
    print("\n[Pipeline 1] Baseline LLM — Direct question to Groq ...")
    p1_result = call_llm_baseline(question)
    print(f"   ✅ Pipeline 1 done in {p1_result['response_time']}s "
          f"| Tokens: {p1_result['total_tokens']}")

    # ── Pipeline 2: GraphRAG ──────────────────────────────────────────────────
    print("\n[Pipeline 2] GraphRAG — Retrieve context from TigerGraph, then Groq ...")

    graph_info = _run_graph_retrieval(question)

    p2_result  = call_llm_with_graph_context(question, graph_info["context_text"])
    print(f"   ✅ Pipeline 2 done in {p2_result['response_time']}s "
          f"| Tokens: {p2_result['total_tokens']} "
          f"| Nodes retrieved: {graph_info['nodes_found']}")

    # ── Metrics Comparison ────────────────────────────────────────────────────
    metrics = compute_metrics(p1_result, p2_result, graph_info)

    print(f"\n{'='*60}")
    print("📊 Results Summary:")
    print(f"   Baseline tokens : {metrics['baseline_total_tokens']}")
    print(f"   GraphRAG tokens : {metrics['graphrag_total_tokens']}")
    print(f"   Context quality : {metrics['context_quality_label']}")
    print(f"{'='*60}\n")

    return {
        "pipeline1":  p1_result,
        "pipeline2":  p2_result,
        "graph_info": graph_info,
        "metrics":    metrics,
        "question":   question,
        "timestamp":  timestamp,
    }


def _run_graph_retrieval(question: str) -> dict:
    """
    Attempt to retrieve context from TigerGraph.
    If TigerGraph is unavailable, falls back to local knowledge base.
    """
    try:
        conn = get_connection()
        return find_relevant_context(conn, question)

    except SystemExit:
        print("   ⚠️  TigerGraph not configured. Using local knowledge fallback.")
        return _local_fallback_context(question)

    except Exception as e:
        print(f"   ⚠️  TigerGraph error: {e}. Using local knowledge fallback.")
        return _local_fallback_context(question)


def _local_fallback_context(question: str) -> dict:
    """
    Fallback: when TigerGraph is not available.
    """
    from data.knowledge import get_all_entities, get_all_relationships
    from graph.query    import _extract_seed_entities

    all_entities = get_all_entities()
    all_rels     = get_all_relationships()

    seed_ids = _extract_seed_entities(question)
    if not seed_ids:
        seed_ids = ["ml_001", "nlp_002", "graph_004"]

    entity_lookup = {e["id"]: e for e in all_entities}

    selected_ids = set(seed_ids)
    for rel in all_rels:
        if rel["source"] in seed_ids:
            selected_ids.add(rel["target"])
        if rel["target"] in seed_ids:
            selected_ids.add(rel["source"])

    selected_ids  = list(selected_ids)[:config.MAX_CONTEXT_NODES]
    selected_ents = [entity_lookup[eid] for eid in selected_ids if eid in entity_lookup]

    selected_rels = [
        r for r in all_rels
        if r["source"] in selected_ids and r["target"] in selected_ids
    ]

    lines = ["ENTITIES FROM KNOWLEDGE BASE (Local Fallback):\n"]
    for i, ent in enumerate(selected_ents, 1):
        lines.append(f"{i}. [{ent['type']}] {ent['name']} (Domain: {ent['domain']})")
        lines.append(f"   {ent['description']}\n")

    if selected_rels:
        lines.append("\nRELATIONSHIPS:\n")
        for rel in selected_rels[:10]:
            lines.append(
                f"• {rel['source']} --[{rel['relation']}]--> {rel['target']}: "
                f"{rel['description']}"
            )

    context_text = "\n".join(lines)

    return {
        "context_text":  context_text,
        "nodes_found":   len(selected_ents),
        "seed_entities": seed_ids,
        "all_entities":  selected_ents,
        "fallback":      True,
        "fallback_note": "⚠️ TigerGraph not connected — using local knowledge base",
    }
