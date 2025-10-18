import time
from tabulate import tabulate
from pipelines.rag_pipeline import rag_pipeline
from pipelines.ref_rag_pipeline import refrag_pipeline
from .metrics import exact_match, f1_score

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

def display_results_table(total_queries, rag_time, rag_em, rag_f1, refrag_time, refrag_em, refrag_f1):
    """Display RAG vs REFRAG results in a clean tabular format"""
    data = [
        ["Total Queries", total_queries, total_queries],
        ["Avg Time (s)", f"{rag_time:.2f}", f"{refrag_time:.2f}"],
        ["Exact Match (EM %)", f"{rag_em:.2f}", f"{refrag_em:.2f}"],
        ["F1 Score (%)", f"{rag_f1:.2f}", f"{refrag_f1:.2f}"]
    ]
    headers = ["Metric", "RAG", "REFRAG"]
    
    print("\n" + "=" * 50)
    print("RAG vs REFRAG Comparison")
    print("=" * 50)
    print(tabulate(data, headers=headers, tablefmt="fancy_grid"))
    print("\nExecution completed successfully!\n")

def run_benchmark(validation_set, retriever, refrag_retriever, llm, max_queries=None):
    """
    Run RAG vs REFRAG on multiple queries from validation set.
    Returns list of results and prints summary statistics.
    """
    results = []
    total_rag_time = 0
    total_refrag_time = 0
    total_rag_em = 0
    total_refrag_em = 0
    total_rag_f1 = 0
    total_refrag_f1 = 0

    queries_to_run = validation_set if max_queries is None else validation_set[:max_queries]

    for i, item in enumerate(queries_to_run, 1):
        result = run_query(item["query"], retriever, refrag_retriever, llm)
        results.append(result)

        # Accumulate times
        total_rag_time += result["rag_time"]
        total_refrag_time += result["refrag_time"]

        # Evaluate answers
        gt = item["ground_truth"]
        total_rag_em += exact_match(result["rag_answer"], gt)
        total_refrag_em += exact_match(result["refrag_answer"], gt)
        total_rag_f1 += f1_score(result["rag_answer"], gt)
        total_refrag_f1 += f1_score(result["refrag_answer"], gt)

    avg_rag_time = total_rag_time / len(results)
    avg_refrag_time = total_refrag_time / len(results)

    n = len(queries_to_run)
    avg_rag_em = total_rag_em / n * 100
    avg_refrag_em = total_refrag_em / n * 100
    avg_rag_f1 = total_rag_f1 / n * 100
    avg_refrag_f1 = total_refrag_f1 / n * 100

    display_results_table(
        total_queries=n,
        rag_time=avg_rag_time,
        rag_em=avg_rag_em,
        rag_f1=avg_rag_f1,
        refrag_time=avg_refrag_time,
        refrag_em=avg_refrag_em,
        refrag_f1=avg_refrag_f1
    )
    return results