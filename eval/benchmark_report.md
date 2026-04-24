# GraphRAG Benchmark Report

## Summary
- Total questions tested: 5
- Average token savings: -265.5%
- Average cost savings: $0.00000000
- Faster Pipeline on average: Pipeline 1 (Baseline)
- Winner: GraphRAG Pipeline

## Results Table
| Question | P1 Tokens | P2 Tokens | Savings | P1 Time | P2 Time |
|----------|-----------|-----------|---------|---------|---------|
| What is machine learning? | 372 | 1240 | -233.3% | 2.153s | 1.338s |
| How does a graph database w... | 461 | 832 | -80.5% | 1.013s | 1.15s |
| What is the difference betw... | 363 | 1316 | -262.5% | 5.709s | 12.389s |
| What is RAG in AI? | 223 | 1370 | -514.3% | 2.881s | 11.798s |
| What is deep learning? | 384 | 1293 | -236.7% | 5.083s | 10.969s |

## Detailed Results
### Q1: What is machine learning?
- **P1 Answer (excerpt):** **Machine Learning (ML) Definition:**  Machine Learning is a subset of Artificial Intelligence (AI) ...
- **P2 Answer (excerpt):** Based on the provided context, machine learning (ML) is a subset of artificial intelligence where sy...
- **Graph Context:** 7 entities found (🟡 Good)
- **Metrics:** P1 Tokens=372, P2 Tokens=1240, P1 Time=2.153s, P2 Time=1.338s

### Q2: How does a graph database work?
- **P1 Answer (excerpt):** A graph database is a type of NoSQL database designed to store and query complex relationships betwe...
- **P2 Answer (excerpt):** A graph database is a type of NoSQL database that stores data as nodes (entities) and edges (relatio...
- **Graph Context:** 3 entities found (🟡 Good)
- **Metrics:** P1 Tokens=461, P2 Tokens=832, P1 Time=1.013s, P2 Time=1.15s

### Q3: What is the difference between AI and ML?
- **P1 Answer (excerpt):** **AI (Artificial Intelligence) vs. ML (Machine Learning):**  While often used interchangeably, AI an...
- **P2 Answer (excerpt):** Based on the provided knowledge graph context, I will explain the difference between Artificial Inte...
- **Graph Context:** 10 entities found (🟢 Excellent)
- **Metrics:** P1 Tokens=363, P2 Tokens=1316, P1 Time=5.709s, P2 Time=12.389s

### Q4: What is RAG in AI?
- **P1 Answer (excerpt):** In the context of Artificial Intelligence (AI), RAG stands for "Reality-Adaptive Generator." However...
- **P2 Answer (excerpt):** Based on the knowledge graph context provided, I can give a detailed and accurate explanation of Ret...
- **Graph Context:** 10 entities found (🟢 Excellent)
- **Metrics:** P1 Tokens=223, P2 Tokens=1370, P1 Time=2.881s, P2 Time=11.798s

### Q5: What is deep learning?
- **P1 Answer (excerpt):** Deep learning is a subset of machine learning, a branch of artificial intelligence (AI) that involve...
- **P2 Answer (excerpt):** Based on the provided knowledge graph context, Deep Learning is a subset of Machine Learning that us...
- **Graph Context:** 7 entities found (🟡 Good)
- **Metrics:** P1 Tokens=384, P2 Tokens=1293, P1 Time=5.083s, P2 Time=10.969s

## Conclusion
GraphRAG wins because:
- Uses -265.5% fewer tokens on average (if context is highly optimized).
- Provides structured context from graph.
- More accurate answers using entity relationships.
