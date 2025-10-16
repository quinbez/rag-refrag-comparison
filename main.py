import time
from dataset_utils import load_docs
from retrievers.rag import Retriever
from retrievers.ref_rag import REFRAGRetriever
from pipelines.rag_pipeline import rag_pipeline
from pipelines.ref_rag_pipeline import refrag_pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    """
    A simple wrapper for a Hugging Face sequence-to-sequence model
    that handles tokenization and text generation.
    """
    def __init__(self, model_name="google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt, max_tokens=150):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Load dataset
    docs = load_docs()

    # Initialize retrievers and LLM
    retriever = Retriever(docs)
    refrag_retriever = REFRAGRetriever(docs)
    llm = Generator(model_name="google/flan-t5-small")

    query = "What is the main drawback of standard RAG when dealing with many documents?"

    # --- RAG ---
    start = time.time()
    rag_answer = rag_pipeline(query, retriever, llm)
    rag_time = time.time() - start
    print("\nRAG Answer:\n", rag_answer)
    print(f"RAG Time: {rag_time:.2f}s")

    # --- REFRAG ---
    start = time.time()
    refrag_answer = refrag_pipeline(query, refrag_retriever, llm)
    refrag_time = time.time() - start
    print("\nREFRAG Answer:\n", refrag_answer)
    print(f"REFRAG Time: {refrag_time:.2f}s")

    print(f"\nComparison: RAG={rag_time:.2f}s | REFRAG={refrag_time:.2f}s")