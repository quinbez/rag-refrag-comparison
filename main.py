import os
import sys
import time
import pickle
from dataset_utils import load_docs
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
            print(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print("    Model loaded successfully!")
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
            print("Loading retrievers from cache...")

            with open(rag_cache, 'rb') as f:
                retriever = pickle.load(f)

            with open(refrag_cache, 'rb') as f:
                refrag_retriever = pickle.load(f)

            print("    Retrievers loaded from cache!")
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


if __name__ == "__main__":
    print("="*80)
    print("RAG vs REFRAG Comparison")
    print("="*80)

    try:
        # Load dataset
        docs = load_docs()
        doc_texts = [f"{doc['title']} {doc['description']}" for doc in docs]

        retriever, refrag_retriever = load_or_create_retrievers(docs)
        llm = Generator(model_name="google/flan-t5-small")

        query = "What is the main drawback of standard RAG when dealing with many documents?"

        result = run_query(query, retriever, refrag_retriever, llm)

        if result:
            print("\n" + "="*80)
            print("RESULTS SUMMARY")
            print("="*80)
            print(f"Query: {result['query']}")
            print(f"\nRAG: {result['rag_answer']}")
            print(f"Time: {result['rag_time']:.2f}s")
            print(f"\nREFRAG: {result['refrag_answer']}")
            print(f"Time: {result['refrag_time']:.2f}s")
        
        print("\n" + "="*80)
        print("Execution completed successfully!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)