import os
import sys
import time
import pickle
from dataset_utils import load_train_docs, load_validation_set
from retrievers.rag import Retriever
from retrievers.ref_rag import REFRAGRetriever
from pipelines.rag_pipeline import rag_pipeline
from pipelines.ref_rag_pipeline import refrag_pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Cache directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class Generator:
    """
    A simple wrapper for a Hugging Face sequence-to-sequence model
    that handles tokenization and text generation.
    """
    def __init__(self, model_name="google/flan-t5-small"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            sys.exit(1)

    def generate(self, prompt, max_tokens=150):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"ERROR in generation: {e}")
            return f"Generation failed: {str(e)}"


def load_or_create_retrievers(docs):
    """Load retrievers from cache or create new ones"""
    rag_cache = os.path.join(CACHE_DIR, "rag_retriever.pkl")
    refrag_cache = os.path.join(CACHE_DIR, "refrag_retriever.pkl")
    
    try:
        if os.path.exists(rag_cache) and os.path.exists(refrag_cache):
            with open(rag_cache, 'rb') as f:
                retriever = pickle.load(f)

            with open(refrag_cache, 'rb') as f:
                refrag_retriever = pickle.load(f)
        else:
            print("Building retrievers (this will take a moment)...")
            doc_texts = [f"{doc['title']} {doc['description']}" for doc in docs]
            
            retriever = Retriever(docs, doc_texts)
            refrag_retriever = REFRAGRetriever(docs, doc_texts)
            
            # Save to cache
            print("Saving retrievers to cache...")

            with open(rag_cache, 'wb') as f:
                pickle.dump(retriever, f)

            with open(refrag_cache, 'wb') as f:
                pickle.dump(refrag_retriever, f)

            print("    Retrievers cached!")
        
        return retriever, refrag_retriever
    except Exception as e:
        print(f"ERROR: Failed to load/create retrievers: {e}")
        sys.exit(1)


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

def benchmark(validation_set, retriever, refrag_retriever, llm, max_queries=None):
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
        results = benchmark(validation_set, retriever, refrag_retriever, llm)

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