"""
config.py - Central Configuration File
=======================================
Loads all settings from the .env file using python-dotenv.
NEVER hardcode any API keys here - always use .env!
"""

import os
from dotenv import load_dotenv

# Load the .env file into environment variables
load_dotenv()

# ── Groq LLM Settings ────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL      = "llama-3.1-8b-instant"
GROQ_BASE_URL  = "https://api.groq.com"

# Cost estimate: Groq free tier is $0.0
# If you want to put actual pricing, it is very low. We set to 0.0 for now.
GROQ_INPUT_COST_PER_TOKEN  = 0.0
GROQ_OUTPUT_COST_PER_TOKEN = 0.0

# ── TigerGraph Settings ───────────────────────────────────────────────────────
TIGERGRAPH_HOST       = os.environ.get("TIGERGRAPH_HOST", "")
if TIGERGRAPH_HOST and not TIGERGRAPH_HOST.startswith("http"):
    TIGERGRAPH_HOST = "https://" + TIGERGRAPH_HOST
TIGERGRAPH_USERNAME   = os.environ.get("TIGERGRAPH_USERNAME", "tigergraph")
TIGERGRAPH_PASSWORD   = os.environ.get("TIGERGRAPH_PASSWORD", "")
TIGERGRAPH_SECRET     = os.environ.get("TIGERGRAPH_SECRET", "")
TIGERGRAPH_GRAPH_NAME = os.environ.get("TIGERGRAPH_GRAPH_NAME", "GraphRAGDemo")

print("--- TigerGraph Config Debug ---")
print(f"TIGERGRAPH_HOST: {TIGERGRAPH_HOST}")
print(f"TIGERGRAPH_USERNAME: {TIGERGRAPH_USERNAME}")
print(f"TIGERGRAPH_SECRET: {TIGERGRAPH_SECRET[:5] + '***' if TIGERGRAPH_SECRET else 'None'}")
print(f"TIGERGRAPH_GRAPH_NAME: {TIGERGRAPH_GRAPH_NAME}")
print("-------------------------------")

# ── Application Settings ──────────────────────────────────────────────────────
MAX_HOPS            = 2    # How many hops to traverse in the graph
MAX_CONTEXT_NODES   = 10   # Max nodes to pull into LLM context
QUERY_HISTORY_LIMIT = 5    # Show last N queries in the dashboard


def validate_config():
    """
    Check that the required environment variables are set.
    Prints a warning if any are missing.
    Returns True if all good, False if something is missing.
    """
    missing = []
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        missing.append("GROQ_API_KEY")
    if not TIGERGRAPH_HOST or TIGERGRAPH_HOST == "your_tigergraph_host_here":
        missing.append("TIGERGRAPH_HOST")
    if not TIGERGRAPH_PASSWORD or TIGERGRAPH_PASSWORD == "your_tigergraph_password_here":
        missing.append("TIGERGRAPH_PASSWORD")

    if missing:
        print(f"⚠️  WARNING: Missing or placeholder values in .env: {missing}")
        print("   Please update your .env file with real credentials.")
        return False

    print("✅ Config loaded successfully.")
    return True
