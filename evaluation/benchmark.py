import time
from pipelines.rag_pipeline import rag_pipeline
from pipelines.ref_rag_pipeline import refrag_pipeline

def run_query(query, retriever, refrag_retriever, llm):
    """Run a query with error handling"""
    
    if not query or not query.strip():
        print("ERROR: Empty query provided")
        return None, None
    try:
        # RAG
        start = time.time()
        rag_answer = rag_pipeline(query, retriever, llm)
        rag_time = time.time() - start
    except Exception as e:
        print(f"ERROR in RAG pipeline: {e}")
        rag_answer = f"RAG Error: {str(e)}"
        rag_time = 0.0
    
    try:
        # REFRAG
        start = time.time()
        refrag_answer = refrag_pipeline(query, refrag_retriever, llm)
        refrag_time = time.time() - start
    except Exception as e:
        print(f"ERROR in REFRAG pipeline: {e}")
        refrag_answer = f"REFRAG Error: {str(e)}"
        refrag_time = 0.0
    
    return {
        'query': query,
        'rag_answer': rag_answer,
        'rag_time': rag_time,
        'refrag_answer': refrag_answer,
        'refrag_time': refrag_time
    }

def run_benchmark(validation_set, retriever, refrag_retriever, llm, max_queries=None):
    """
    Run RAG vs REFRAG on multiple queries from validation set.
    Returns list of results and prints summary statistics.
    """
    results = []
    total_rag_time = 0
    total_refrag_time = 0

    queries_to_run = validation_set if max_queries is None else validation_set[:max_queries]

    print(f"\nRunning benchmark on {len(queries_to_run)} queries...\n")

    for i, item in enumerate(queries_to_run, 1):
        print(f"Query {i}/{len(queries_to_run)}: {item['query'][:60]}...")
        result = run_query(item["query"], retriever, refrag_retriever, llm)
        results.append(result)
        total_rag_time += result["rag_time"]
        total_refrag_time += result["refrag_time"]

    avg_rag_time = total_rag_time / len(results)
    avg_refrag_time = total_refrag_time / len(results)

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total queries run: {len(results)}")
    print(f"Average RAG time per query: {avg_rag_time:.2f}s")
    print(f"Average REFRAG time per query: {avg_refrag_time:.2f}s")

    return results