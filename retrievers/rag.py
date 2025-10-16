import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    Retriever for semantic search over a set of documents using FAISS.

    Args:
        docs (List[str]): List of documents to index for retrieval.
        embedding_model (str): Name of the SentenceTransformer model to convert
                               documents and queries into dense embeddings.

    Attributes:
        docs (List[str]): The original documents.
        embedder (SentenceTransformer): Model to compute embeddings.
        embeddings (np.ndarray): Dense vector representations of documents.
        index (faiss.IndexFlatL2): FAISS index for fast similarity search.
    """
    def __init__(self, docs: List[str], embedding_model: str = "all-MiniLM-L6-v2"):
        self.docs = docs
        self.embedder = SentenceTransformer(embedding_model)
        
        # Encode documents to embeddings
        self.embeddings = self.embedder.encode(docs, convert_to_tensor=True).cpu().numpy()
        
        # Initialize FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
      
    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Perform a semantic search for the top-k most similar documents to the query.

        Args:
            query (str): The query string to search for.
            k (int): Number of top documents to return. Defaults to 3.

        Returns:
            List[str]: List of top-k retrieved documents.
        """
        query_emb = self.embedder.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(query_emb, k)
        
        return [self.docs[i] for i in indices[0]]
