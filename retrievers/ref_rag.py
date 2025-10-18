from typing import List, Tuple, Union, Dict
from retrievers.rag import Retriever

class REFRAGRetriever(Retriever):
    """
    An enhanced retriever that extends the base Retriever class.
    It re-ranks initially retrieved documents based on inverse-distance scores
    and returns the top-k most relevant documents along with their weights.

    Enhanced retriever with score filtering and diversity.

    Attributes:
        Inherits all attributes from the `Retriever` class.
        score_threshold (float): Minimum score to include a document (default: 0.5)
        diversity_threshold (float): Minimum difference between documents (default: 0.3)
    """
    def __init__(self, docs, doc_texts=None, embedding_model="all-MiniLM-L6-v2", 
                 score_threshold=0.5, diversity_threshold=0.3):
        super().__init__(docs, doc_texts, embedding_model)
        self.score_threshold = score_threshold
        self.diversity_threshold = diversity_threshold
        
    def search(self, query: str, k: int = 3) -> List[Tuple[Union[Dict, str], float]]:
        """
        Searches for documents relevant to a given query and diversity, and re-ranks them
        based on inverse-distance scores.

        Args:
            query (str): The input query string.
            k (int): Number of top documents to return. Defaults to 3.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing (document, relevance_score),
                                     sorted by descending score.
        """
        query_emb = self.embedder.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(query_emb, k * 3)
        distances, indices = distances[0], indices[0]

        # Compute inverse-distance scores 
        scores = 1.0 / (distances + 1e-5)

        # Normalize scores to 0-1 range
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        # Filter by relevance threshold
        filtered_docs = []
        for idx, i in enumerate(indices):
            if scores[idx] >= self.score_threshold:
                filtered_docs.append((self.docs[i], float(scores[idx])))

        # Apply diversity filtering 
        diverse_docs = []
        for doc, score in filtered_docs:
            is_diverse = True
            doc_text = f"{doc.get('title', '')} {doc.get('description', '')}"
            
            for existing_doc, _ in diverse_docs:
                existing_text = f"{existing_doc.get('title', '')} {existing_doc.get('description', '')}"

                doc_words = set(doc_text.lower().split())
                existing_words = set(existing_text.lower().split())
                
                if len(doc_words) > 0 and len(existing_words) > 0:
                    overlap = len(doc_words & existing_words) / max(len(doc_words), len(existing_words))
                    if overlap > (1 - self.diversity_threshold):
                        is_diverse = False
                        break
        
            if is_diverse:
                    diverse_docs.append((doc, score))
                
            if len(diverse_docs) >= k:
                break
        
        return diverse_docs[:k]