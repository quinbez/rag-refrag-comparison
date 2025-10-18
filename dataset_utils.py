import os
import pickle
from datasets import load_dataset

# Cache directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_docs(force_reload=False):
    """
    Loads documents from the TriviaQA dataset and extracts relevant 
    fields for retrieval.

    Returns:
        List[Dict[str, str]]: A list of document dictionaries containing:
            - title
            - url
            - description
            - filename
            - rank
            - search_context

    Raises:
        ValueError: If no valid documents are found in the dataset.
    """
    cache_file = os.path.join(CACHE_DIR, "trivia_qa_docs.pkl")

    # Try to load from cache
    if not force_reload and os.path.exists(cache_file):
        print(f"Loading documents from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            docs = pickle.load(f)
        print(f"    Loaded {len(docs)} documents from cache!")
        return docs
    
    print("Loading dataset from HuggingFace (this may take a while)...")
    dataset = load_dataset("trivia_qa", "rc.web")
    sample = dataset["train"][0]
    search_results = sample["search_results"]

    docs = []
    for i in range(len(search_results["title"])):
        doc = {
            "title": search_results["title"][i],
            "url": search_results["url"][i],    
            "description": search_results["description"][i],
            "filename": search_results["filename"][i],
            "rank": search_results["rank"][i],
            "search_context": search_results["search_context"][i],
        }
        docs.append(doc)

    print(f"Number of documents extracted: {len(docs)}")
    
    if len(docs) == 0:
        raise ValueError("No valid documents found for retrieval!")

    # Save to cache
    print(f"Saving documents to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(docs, f)

    return docs