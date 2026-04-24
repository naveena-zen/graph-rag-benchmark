"""
eval/metrics.py - Evaluation Metrics Calculator
================================================
Computes comparison metrics between Pipeline 1 (Baseline LLM)
and Pipeline 2 (GraphRAG) results.

Metrics include:
  - Token usage comparison
  - Response time comparison
  - Cost comparison in USD
  - Context quality score (for GraphRAG)
  - Token savings / overhead calculation
"""


def compute_metrics(baseline_result: dict, graphrag_result: dict, graph_context_info: dict) -> dict:
    """
    Compute comparison metrics between baseline and GraphRAG pipelines.

    Args:
        baseline_result   : Output dict from llm/caller.py for Pipeline 1
        graphrag_result   : Output dict from llm/caller.py for Pipeline 2
        graph_context_info: Output dict from graph/query.py

    Returns:
        dict with all comparison metrics, ready for the dashboard.
    """

    # ── Token Metrics ─────────────────────────────────────────────────────────
    baseline_tokens = baseline_result.get("total_tokens",  0)
    graphrag_tokens = graphrag_result.get("total_tokens",  0)
    token_diff      = graphrag_tokens - baseline_tokens
    token_overhead  = (
        f"+{token_diff} (GraphRAG uses more tokens due to context)"
        if token_diff >= 0
        else f"{token_diff} (GraphRAG used fewer tokens)"
    )

    # ── Time Metrics ──────────────────────────────────────────────────────────
    baseline_time = baseline_result.get("response_time", 0.0)
    graphrag_time = graphrag_result.get("response_time", 0.0)
    time_diff     = graphrag_time - baseline_time
    faster_slower = "faster" if time_diff < 0 else "slower"

    # ── Cost Metrics ──────────────────────────────────────────────────────────
    baseline_cost = baseline_result.get("cost_usd", 0.0)
    graphrag_cost = graphrag_result.get("cost_usd", 0.0)
    cost_diff     = graphrag_cost - baseline_cost

    # ── Context Quality Score (GraphRAG only) ─────────────────────────────────
    # Simple heuristic score based on:
    #   - Number of nodes found (more = better context coverage)
    #   - Whether seed entities were found (keyword relevance)
    nodes_found   = graph_context_info.get("nodes_found",   0)
    seed_entities = graph_context_info.get("seed_entities", [])

    context_quality = _compute_context_quality(nodes_found, seed_entities)

    # ── Answer Length ─────────────────────────────────────────────────────────
    baseline_len = len(baseline_result.get("answer", "").split())
    graphrag_len = len(graphrag_result.get("answer", "").split())

    # ── Summary ───────────────────────────────────────────────────────────────
    return {
        # Token comparison
        "baseline_input_tokens":  baseline_result.get("input_tokens",  0),
        "baseline_output_tokens": baseline_result.get("output_tokens", 0),
        "baseline_total_tokens":  baseline_tokens,

        "graphrag_input_tokens":  graphrag_result.get("input_tokens",  0),
        "graphrag_output_tokens": graphrag_result.get("output_tokens", 0),
        "graphrag_total_tokens":  graphrag_tokens,

        "token_difference":       token_diff,
        "token_overhead_note":    token_overhead,

        # Time comparison (seconds)
        "baseline_response_time": baseline_time,
        "graphrag_response_time": graphrag_time,
        "time_difference":        round(time_diff, 3),
        "time_note":              f"GraphRAG is {abs(round(time_diff,2))}s {faster_slower}",

        # Cost comparison (USD)
        "baseline_cost_usd":      baseline_cost,
        "graphrag_cost_usd":      graphrag_cost,
        "cost_difference_usd":    round(cost_diff, 8),

        # Context quality
        "context_quality_score":  context_quality,
        "context_quality_label":  _quality_label(context_quality),
        "nodes_retrieved":        nodes_found,
        "seed_entities_found":    len(seed_entities),

        # Answer richness
        "baseline_word_count":    baseline_len,
        "graphrag_word_count":    graphrag_len,

        # Model info
        "model_used":             baseline_result.get("model", "llama-3.1-8b-instant"),

        # Error flags
        "baseline_error":         baseline_result.get("error"),
        "graphrag_error":         graphrag_result.get("error"),
    }


def _compute_context_quality(nodes_found: int, seed_entities: list) -> float:
    """
    Compute a 0.0–1.0 context quality score for GraphRAG.

    Formula:
      - 0.5 base if any seed entities were matched
      - +0.1 per additional seed entity (up to 0.3)
      - +0.1 per 2 nodes found (up to 0.2)
    """
    score = 0.0

    if seed_entities:
        score += 0.5
        score += min(0.3, len(seed_entities) * 0.1)

    score += min(0.2, (nodes_found // 2) * 0.05)

    return round(min(1.0, score), 2)


def _quality_label(score: float) -> str:
    """Convert quality score to a human-friendly label."""
    if score >= 0.8:
        return "🟢 Excellent"
    elif score >= 0.6:
        return "🟡 Good"
    elif score >= 0.4:
        return "🟠 Fair"
    elif score > 0.0:
        return "🔴 Low"
    else:
        return "⚪ None"


def format_cost(cost_usd: float) -> str:
    """Format a cost in USD as a readable string."""
    if cost_usd < 0.000001:
        return "< $0.000001"
    return f"${cost_usd:.8f}"


def format_time(seconds: float) -> str:
    """Format seconds as a readable time string."""
    return f"{seconds:.2f}s"
