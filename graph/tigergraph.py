from graph.graph import get_connection, _local_fallback, extract_seed_entities

def get_local_context(question):
    # Use existing fallback logic
    seeds = extract_seed_entities(question)
    res = _local_fallback(question, seeds)
    return res.get("context_text", "")

def extract_keywords(question):
    # Simple keyword extraction
    words = [w.lower() for w in question.replace("?", "").replace(".", "").replace(",", "").split()]
    return [w for w in words if len(w) > 4]

TOP_K = 3
NUM_HOPS = 2
NUM_SEEN_MIN = 2

def get_graph_context(question):
    conn = get_connection()
    if not conn:
        return get_local_context(question)
    
    try:
        keywords = extract_keywords(question)
        # print(f"Keywords: {keywords}")
        
        seed_entities = []
        for keyword in keywords[:TOP_K]:
            try:
                results = conn.getVertices(
                    "Entity",
                    where=f"name LIKE \"%{keyword}%\""
                )
                seed_entities.extend(results)
            except:
                pass
        
        # print(f"Seed entities: {len(seed_entities)}")
        
        context_parts = []
        visited = set()
        
        for entity in seed_entities[:TOP_K]:
            entity_id = entity.get("v_id","")
            if entity_id in visited:
                continue
            visited.add(entity_id)
            
            attrs = entity.get("attributes",{})
            name = attrs.get("name", entity_id)
            desc = attrs.get("description","")
            
            if name and desc:
                context_parts.append(
                    f"{name}: {desc}")
            
            try:
                neighbors = conn.getVertexNeighbors(
                    "Entity", entity_id,
                    edgeType="RELATED_TO",
                    maxSize=NUM_SEEN_MIN
                )
                for n in neighbors[:2]:
                    n_id = n.get("v_id","")
                    if n_id not in visited:
                        visited.add(n_id)
                        n_attrs = n.get(
                            "attributes",{})
                        n_name = n_attrs.get(
                            "name","")
                        n_desc = n_attrs.get(
                            "description","")
                        if n_name:
                            context_parts.append(
                                f"{n_name}: {n_desc}")
            except Exception as e:
                pass # print(f"Neighbor error: {e}")
        
        if context_parts:
            context = "\n".join(context_parts[:5])
            words = context.split()[:80]
            return " ".join(words)
        
        return get_local_context(question)
        
    except Exception as e:
        # print(f"TigerGraph query error: {e}")
        return get_local_context(question)
