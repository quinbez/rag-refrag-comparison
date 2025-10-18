import sys
from dataset_utils import load_train_docs, load_validation_set
from retrievers.generator import Generator
from evaluation.benchmark import run_benchmark
from retrievers.utils import load_or_create_retrievers
from evaluation.visualize import plot_rag_vs_refrag

if __name__ == "__main__":
    try:
        # Load dataset
        docs = load_train_docs()
        validation_set = load_validation_set(num_samples=50) 
        retriever, refrag_retriever = load_or_create_retrievers(docs)
        llm = Generator(model_name="google/flan-t5-small")

        # Run benchmark
        results = run_benchmark(validation_set, retriever, refrag_retriever, llm)

        # Visualize
        plot_rag_vs_refrag(results)
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)