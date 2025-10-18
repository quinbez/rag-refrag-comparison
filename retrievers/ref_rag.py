from typing import List, Tuple, Union, Dict
from retrievers.rag import Retriever

class REFRAGRetriever(Retriever):
    """
    An enhanced retriever that extends the base Retriever class.
    It re-ranks initially retrieved documents based on inverse-distance scores
    and returns the top-k most relevant documents along with their weights.

    Attributes:
        Inherits all attributes from the `Retriever` class.
    """

    def search(self, query: str, k: int = 3) -> List[Tuple[Union[Dict, str], float]]:
        """
        Searches for documents relevant to a given query and re-ranks them
        based on inverse-distance scores.

        Args:
            query (str): The input query string.
            k (int): Number of top documents to return. Defaults to 3.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing (document, relevance_score),
                                     sorted by descending score.
        """
        query_emb = self.embedder.encode([query], convert_to_tensor=False)

        distances, indices = self.index.search(query_emb, k * 2)
        distances, indices = distances[0], indices[0]

        # Compute inverse-distance scores 
        scores = 1.0 / (distances + 1e-5)

        docs_with_scores = [(self.docs[i], float(scores[idx])) for idx, i in enumerate(indices)]
        top_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)[:k]

        return top_docs