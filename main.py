"""
main.py - Entry Point
======================
"""

import sys
import os

# This adds the graphrag_project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def main():
    print("=" * 65)
    print("  🧠 GraphRAG Inference System — TigerGraph Hackathon")
    print("=" * 65)

    # ── Step 1: Validate Configuration ────────────────────────────────────────
    print("\n📋 Step 1: Validating configuration ...")
    config_ok = config.validate_config()

    if not config_ok:
        print("\n⚠️  Some credentials are missing. The system will still run")
        print("   with a LOCAL FALLBACK for TigerGraph (no real DB needed).")
        print("   The pipeline requires a valid GROQ_API_KEY.\n")

    # ── Step 2: Attempt TigerGraph Setup ─────────────────────────────────────
    tg_available = False

    if (
        config.TIGERGRAPH_HOST and
        config.TIGERGRAPH_HOST != "your_tigergraph_host_here" and
        config.TIGERGRAPH_PASSWORD and
        config.TIGERGRAPH_PASSWORD != "your_tigergraph_password_here"
    ):
        print("\n🔌 Step 2: Connecting to TigerGraph ...")
        try:
            from graph.connection import get_connection
            from graph.schema     import create_schema
            from graph.loader     import load_data, check_data_loaded

            conn = get_connection()

            print("\n📐 Step 3: Creating graph schema ...")
            create_schema(conn)

            print("\n📥 Step 4: Loading knowledge data ...")
            if not check_data_loaded(conn):
                result = load_data(conn)
                print(f"   Loaded: {result['vertices_loaded']} vertices, "
                      f"{result['edges_loaded']} edges")
            else:
                print("   Data already loaded — skipping.")

            tg_available = True

        except Exception as e:
            print(f"\n⚠️  TigerGraph setup failed: {e}")
            print("   Falling back to local knowledge base for demo.")
    else:
        print("\n⏭️  Step 2-4: Skipping TigerGraph setup (credentials not set).")
        print("   Using local knowledge base fallback.\n")

    # ── Step 5: Test Pipeline ─────────────────────────────────────────────────
    if (
        config.GROQ_API_KEY and
        config.GROQ_API_KEY != "your_groq_api_key_here"
    ):
        print("\n🧪 Step 5: Running pipeline test with sample question ...")
        print("-" * 50)

        try:
            from inference.orchestrator import run_both_pipelines
            result = run_both_pipelines("What is machine learning?")

            print("\n📝 Pipeline 1 Answer (first 200 chars):")
            print("   " + result["pipeline1"]["answer"][:200] + "...")

            print("\n📝 Pipeline 2 Answer (first 200 chars):")
            print("   " + result["pipeline2"]["answer"][:200] + "...")

            print("\n📊 Metrics:")
            m = result["metrics"]
            print(f"   Baseline tokens : {m['baseline_total_tokens']}")
            print(f"   GraphRAG tokens : {m['graphrag_total_tokens']}")
            print(f"   Baseline time   : {m['baseline_response_time']}s")
            print(f"   GraphRAG time   : {m['graphrag_response_time']}s")
            print(f"   Context quality : {m['context_quality_label']}")
            print(f"   Nodes retrieved : {m['nodes_retrieved']}")

        except Exception as e:
            print(f"⚠️  Pipeline test failed: {e}")
            print("   This is OK — the dashboard will handle errors gracefully.")
    else:
        print("\n⏭️  Step 5: Skipping pipeline test (GROQ_API_KEY not set).")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅ Setup complete!")
    print("\n🚀 To start the dashboard, run:")
    print("   streamlit run dashboard/app.py")
    print("\n📝 Fill in your .env file with:")
    print("   1. GROQ_API_KEY        → Your Groq API key")
    print("   2. TIGERGRAPH_HOST     → Your TigerGraph Cloud cluster URL")
    print("   3. TIGERGRAPH_PASSWORD → Your TigerGraph Cloud password")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
