import torch
import os
import faiss
import numpy as np
from typing import List, Tuple, Dict
from retrievers.token_encoder import TokenLevelEncoder
from retrievers.chunk_compressor import ChunkCompressor
from retrievers.relevance_policy import RelevancePolicy
from sentence_transformers import SentenceTransformer
class REFRAGRetriever:
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
    def __init__(
            self,
            docs,
            doc_texts=None,
            embedding_model="all-MiniLM-L6-v2",
            token_model="sentence-transformers/all-MiniLM-L6-v2",
            compressed_dim=128,
            use_pretrained_policy=False,
            policy_path=None,
            score_threshold=0.5,
            diversity_threshold=0.3
        ):
        self.docs = docs
        self.score_threshold = score_threshold
        self.diversity_threshold = diversity_threshold

        texts_to_embed = doc_texts if doc_texts is not None else docs

        # Initialize models
        self.sentence_embedder = SentenceTransformer(embedding_model)
        self.token_encoder = TokenLevelEncoder(token_model)
        self.compressor = ChunkCompressor(input_dim=384, output_dim=compressed_dim)
        self.relevance_policy = RelevancePolicy(chunk_dim=compressed_dim, query_dim=compressed_dim)

        if use_pretrained_policy and policy_path:
            try:
                self.relevance_policy.load_state_dict(torch.load(policy_path))
            except Exception as e:
                print(f"Warning: Could not load policy from {policy_path}: {e}")

        self.relevance_policy.eval()

        print("Computing multi-level embeddings for documents...")
        self._precompute_embeddings(texts_to_embed)

    def _precompute_embeddings(self, texts: List[str]):
        """Precompute both sentence-level and compressed token-level embeddings."""
        # Check for cached embeddings
        if os.path.exists("sentence_embeds.npy") and os.path.exists("compressed_embeds.npy"):
            self.sentence_embeddings = np.load("sentence_embeds.npy")
            self.compressed_embeddings = np.load("compressed_embeds.npy")
            print("Loaded cached embeddings.")

            # Build FAISS index from cached sentence embeddings
            dim = self.sentence_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.sentence_embeddings)
            print("FAISS index built from cached embeddings.")
            return

        # Compute sentence embeddings
        self.sentence_embeddings = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embs = self.sentence_embedder.encode(
                batch,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            self.sentence_embeddings.extend(batch_embs)

        self.sentence_embeddings = np.asarray(self.sentence_embeddings, dtype=np.float32)

        # Token-level compressed embeddings
        self.compressed_embeddings = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"  Processing document {i}/{len(texts)}...")

            token_embs = self.token_encoder.encode(text)
            with torch.no_grad():
                compressed = self.compressor(token_embs)
            self.compressed_embeddings.append(compressed.cpu().numpy())

        self.compressed_embeddings = np.array(self.compressed_embeddings)

        # Save embeddings
        np.save("sentence_embeds.npy", self.sentence_embeddings)
        np.save("compressed_embeds.npy", self.compressed_embeddings)

        # Build FAISS index on sentence embeddings
        dim = self.sentence_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.sentence_embeddings)

        print("Embeddings computed and indexed!")

    def search(self, query: str, k: int = 3, alpha: float = 0.5) -> List[Tuple[Dict, float]]:
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
        query_sentence_emb = self.sentence_embedder.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(query_sentence_emb, k * 5)
        distances, indices = distances[0], indices[0]

        # Get token-level query embedding
        query_token_emb = self.token_encoder.encode(query)
        with torch.no_grad():
            query_compressed = self.compressor(query_token_emb)

        scored_docs = []

        for idx, dist in zip(indices, distances):
            # Compute inverse-distance scores
            sentence_score = 1.0 / (dist + 1e-5)
            doc_compressed = torch.from_numpy(self.compressed_embeddings[idx])

            with torch.no_grad():
                policy_score = self.relevance_policy(
                    query_compressed,
                    doc_compressed
                ).item()

            # Combined score
            final_score = alpha * sentence_score + (1 - alpha) * policy_score

            scored_docs.append((idx, final_score, policy_score, sentence_score))

        # Normalize scores to 0-1 range
        scores = np.array([score for _, score, _, _ in scored_docs])
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        # Sort by final score
        sorted_docs = sorted(
            zip(indices, scores, [p for _, _, p, _ in scored_docs]),
            key=lambda x: x[1],
            reverse=True
        )

        # Apply diversity filtering
        final_results = []
        for idx, score, policy_score in sorted_docs:
            if score < self.score_threshold:
                continue

            doc = self.docs[idx]
            if self._is_diverse(doc, final_results):
                final_results.append((doc, float(score)))

            if len(final_results) >= k:
                break

        return final_results

    def _is_diverse(
        self,
        doc: Dict,
        existing: List[Tuple[Dict, float]],
    ) -> bool:
        """Check if document is diverse enough from existing results."""
        if not existing:
            return True

        doc_text = f"{doc.get('title', '')} {doc.get('description', '')}"
        doc_words = set(doc_text.lower().split())

        for existing_doc, _ in existing:
            existing_text = f"{existing_doc.get('title', '')} {existing_doc.get('description', '')}"
            existing_words = set(existing_text.lower().split())

            if len(doc_words) > 0 and len(existing_words) > 0:
                overlap = len(doc_words & existing_words) / max(len(doc_words), len(existing_words))
                if overlap > self.diversity_threshold:
                    return False

        return True