from typing import Any

def refrag_pipeline(query: str, retriever: Any, llm: Any, top_k: int = 3) -> str:
    """
    RAG pipeline using a REFRAG retriever that includes document relevance scores.
    Retrieves top-k documents, incorporates their scores into context, and generates
    an answer with a language model.

    Args:
        query (str): The user's input question.
        retriever (Any): An instance of REFRAGRetriever with `search(query, k)` returning (doc, score).
        llm (Any): Language model instance with a `.generate(prompt)` method.
        top_k (int): Number of top documents to retrieve. Defaults to 3.

    Returns:
        str: Generated answer from the language model based on scored context.
    """
    docs_with_scores = retriever.search(query, k=top_k)

    # Build context string with weighted importance for each document
    context_passages = [
        f"[Score: {score:.2f}] {doc.get('title', '')}: {doc.get('description', '')}"
        for doc, score in docs_with_scores
    ]

    context_string = "\n\n---\n\n".join(context_passages)

    prompt = f"""Based on the following context (with relevance scores), provide a clear and concise answer to the user's question.

CONTEXT:
{context_string}

QUESTION:
{query}

ANSWER:
"""
    
    return llm.generate(prompt)