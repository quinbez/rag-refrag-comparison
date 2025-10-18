import os
import matplotlib.pyplot as plt

def plot_rag_vs_refrag(results, save_path="evaluation/plots"):
    """
    Visualize benchmark results for RAG vs REFRAG.
    
    Args:
        results (list of dict): Each dict should have 'rag_time', 'refrag_time',
                                'rag_answer', 'refrag_answer', and metrics if desired.
        save_path (str): Directory to save plots.
    """
    os.makedirs(save_path, exist_ok=True)
    
    n_queries = len(results)
    rag_times = [r["rag_time"] for r in results]
    refrag_times = [r["refrag_time"] for r in results]

    rag_em = [r.get("rag_em", 0) for r in results]
    refrag_em = [r.get("refrag_em", 0) for r in results]
    rag_f1 = [r.get("rag_f1", 0) for r in results]
    refrag_f1 = [r.get("refrag_f1", 0) for r in results]

    queries = list(range(1, n_queries + 1))

    # Plot times
    plt.figure(figsize=(10, 5))
    plt.bar([q - 0.15 for q in queries], rag_times, width=0.3, label="RAG Time (s)")
    plt.bar([q + 0.15 for q in queries], refrag_times, width=0.3, label="REFRAG Time (s)")
    plt.xlabel("Query #")
    plt.ylabel("Time (s)")
    plt.title("RAG vs REFRAG Query Times")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "times.png"))
    plt.close()

    # Plot F1
    plt.figure(figsize=(10, 5))
    plt.bar([q - 0.15 for q in queries], rag_f1, width=0.3, label="RAG F1")
    plt.bar([q + 0.15 for q in queries], refrag_f1, width=0.3, label="REFRAG F1")
    plt.xlabel("Query #")
    plt.ylabel("F1 Score")
    plt.title("RAG vs REFRAG F1 per Query")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "f1.png"))
    plt.close()

    print(f"Plots saved to {save_path}")
