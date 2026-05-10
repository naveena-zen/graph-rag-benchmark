"""
config.py — Central Configuration File
Loads all settings from the .env file using python-dotenv.
NEVER hardcode any API keys here - always use .env!
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL     = "llama-3.1-8b-instant"

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── TigerGraph ────────────────────────────────────────────────────────────────
TIGERGRAPH_HOST       = os.environ.get("TIGERGRAPH_HOST", "")
TIGERGRAPH_USERNAME   = os.environ.get("TIGERGRAPH_USERNAME", "tigergraph")
TIGERGRAPH_PASSWORD   = os.environ.get("TIGERGRAPH_PASSWORD", "")
TIGERGRAPH_SECRET     = os.environ.get("TIGERGRAPH_SECRET", "")
TIGERGRAPH_GRAPH_NAME = os.environ.get("TIGERGRAPH_GRAPH_NAME", "GraphRAGDemo")

if TIGERGRAPH_HOST and not TIGERGRAPH_HOST.startswith("http"):
    TIGERGRAPH_HOST = "https://" + TIGERGRAPH_HOST

# ── Application ───────────────────────────────────────────────────────────────
MAX_HOPS          = 2
MAX_CONTEXT_NODES = 3
_P2_CHUNK_TOKENS  = 256
_P2_TOP_K         = 3
_P3_MAX_TOKENS    = 150
_CHARS_PER_TOKEN  = 4
