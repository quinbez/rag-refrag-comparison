from typing import Any

def rag_pipeline(query: str, retriever: Any, llm: Any, top_k: int = 3) -> str:
    """
    RAG pipeline that retrieves relevant documents
    for a given query and generates an answer using a language model.

    Args:
        query (str): The user's input question.
        retriever (Any): An instance of a Retriever that supports `search(query, k)` method.
        llm (Any): Language model instance with a `.generate(prompt)` method.
        top_k (int): Number of top documents to retrieve and use as context. Defaults to 3.

    Returns:
        str: Generated answer from the language model based on retrieved context.
    """
    retrieved_docs = retriever.search(query, k=top_k)

    context_string = "\n\n---\n\n".join(
        [f"{doc.get('title', '')}: {doc.get('description', '')}" for doc in retrieved_docs]
    )

    prompt = f"""Based on the following context, provide a clear and concise answer to the user's question.

Context:
{context_string}

Question:
{query}
"""

    return llm.generate(prompt)
    