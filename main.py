import os
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
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("    Model loaded!")

    def generate(self, prompt, max_tokens=150):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_or_create_retrievers(docs):
    """Load retrievers from cache or create new ones"""
    rag_cache = os.path.join(CACHE_DIR, "rag_retriever.pkl")
    refrag_cache = os.path.join(CACHE_DIR, "refrag_retriever.pkl")
    
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

if __name__ == "__main__":
    print("===== RAG vs REFRAG Comparison =====")

    # Load dataset
    docs = load_docs()
    doc_texts = [f"{doc['title']} {doc['description']}" for doc in docs]

    retriever, refrag_retriever = load_or_create_retrievers(docs)
    llm = Generator(model_name="google/flan-t5-small")

    query = "What is the main drawback of standard RAG when dealing with many documents?"

    # ---- RAG ----
    start = time.time()
    rag_answer = rag_pipeline(query, retriever, llm)
    rag_time = time.time() - start
    print("\nRAG Answer:\n", rag_answer)
    print(f"RAG Time: {rag_time:.2f}s")

    # ---- REFRAG ----
    start = time.time()
    refrag_answer = refrag_pipeline(query, refrag_retriever, llm)
    refrag_time = time.time() - start
    print("\nREFRAG Answer:\n", refrag_answer)
    print(f"REFRAG Time: {refrag_time:.2f}s")

    print(f"\nComparison: RAG={rag_time:.2f}s | REFRAG={refrag_time:.2f}s")