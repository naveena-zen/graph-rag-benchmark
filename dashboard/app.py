"""
dashboard/app.py — Entry point for `streamlit run dashboard/app.py`

Adds the project root to sys.path and changes cwd so all relative imports
(config, rag, eval, graph, data) resolve correctly regardless of where
the command is run from.
"""
import sys
import os

# ── Resolve project root (one level up from this file) ───────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prepend root to sys.path so `import config`, `from rag.pipelines import ...`
# etc. all work exactly as if running from the project root.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Change cwd so .env, data/, faiss_index.bin etc. are found by relative paths
os.chdir(_ROOT)

# ── Run the real dashboard ────────────────────────────────────────────────────
# Import and exec the main app using runpy so __file__ is set correctly and
# Streamlit page-config fires exactly once.
import runpy
runpy.run_path(os.path.join(_ROOT, "app.py"), run_name="__main__")
