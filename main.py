import sys
from dataset_utils import load_train_docs, load_validation_set
from retrievers.generator import Generator
from evaluation.benchmark import run_benchmark
from retrievers.utils import load_or_create_retrievers

if __name__ == "__main__":
    print("="*80)
    print("RAG vs REFRAG Comparison")
    print("="*80)

    try:
        # Load dataset
        docs = load_train_docs()
        validation_set = load_validation_set(num_samples=5) 

        doc_texts = [f"{doc['title']} {doc['description']}" for doc in docs]

        retriever, refrag_retriever = load_or_create_retrievers(docs)
        llm = Generator(model_name="google/flan-t5-small")

        query = "What is the main drawback of standard RAG when dealing with many documents?"

        # Run benchmark
        results = run_benchmark(validation_set, retriever, refrag_retriever, llm)

        for res in results:
            print("\n" + "-"*60)
            print(f"Query: {res['query']}")
            print(f"RAG Answer: {res['rag_answer']} (Time: {res['rag_time']:.2f}s)")
            print(f"REFRAG Answer: {res['refrag_answer']} (Time: {res['refrag_time']:.2f}s)")
            print("-"*60)
        
        print("Execution completed successfully!")

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)