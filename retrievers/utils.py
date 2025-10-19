import os
import sys
import pickle
from .rag import Retriever
from .ref_rag import REFRAGRetriever

# Cache directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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
            refrag_retriever = REFRAGRetriever(
                docs, 
                doc_texts,
                token_model="sentence-transformers/all-MiniLM-L6-v2",
                compressed_dim=128,
                use_pretrained_policy=False,
                policy_path=None,
                score_threshold=0.5,
                diversity_threshold=0.7
            )
            
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