import pyTigerGraph as tg
from config import (TIGERGRAPH_HOST, 
                    TIGERGRAPH_USERNAME,
                    TIGERGRAPH_PASSWORD,
                    TIGERGRAPH_GRAPH_NAME)

def get_connection():
    try:
        host = TIGERGRAPH_HOST
        if not host.startswith("https://"):
            host = "https://" + host
        
        conn = tg.TigerGraphConnection(
            host=host,
            graphname=TIGERGRAPH_GRAPH_NAME,
            username=TIGERGRAPH_USERNAME,
            password=TIGERGRAPH_PASSWORD
        )
        # Using password, so we just getToken with the password 
        # (or createSecret() if the user is a superuser)
        # We will try a simple getToken or echo
        echo = conn.echo()
        print(f"SUCCESS: Connected to TigerGraph! Echo: {echo}")
        return conn
    except Exception as e:
        print(f"FAILED: TigerGraph connection error: {e}")
        return None

def test_connection():
    conn = get_connection()
    if conn:
        print("TigerGraph connection: OK")
        return True
    print("TigerGraph connection: FAILED")
    return False
