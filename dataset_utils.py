import os
import pickle
from datasets import load_dataset

# Cache directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_train_docs(force_reload=False):
    """
    Load all documents from the TriviaQA training set for retrieval.

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
    
    print("Loading TriviaQA training set from HuggingFace (this may take a while)...")
    dataset = load_dataset("trivia_qa", "rc.web", split="train")
    
    docs = []
    for sample in dataset:
        search_results = sample["search_results"]
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

    print(f"Total training documents extracted: {len(docs)}")
    
    if len(docs) == 0:
        raise ValueError("No valid documents found for retrieval!")

    # Save to cache
    print(f"Saving documents to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(docs, f)

    return docs

def load_validation_set(num_samples=None, force_reload=False):
    """
    Load TriviaQA validation set with ground truth answers.
    
    Args:
        num_samples (int or None): Limit number of test samples; None loads all.
    
    Returns:
        List[Dict]: Each dict contains:
            - query: The question
            - ground_truth: List of acceptable answers
            - ground_truth_str: Primary answer
            - search_results: Associated documents
    """
    cache_file = os.path.join(CACHE_DIR, f"trivia_qa_validation_{num_samples or 'all'}.pkl")
    
    if not force_reload and os.path.exists(cache_file):
        print(f"Loading validation set from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    print("Loading TriviaQA validation set from HuggingFace...")
    dataset = load_dataset("trivia_qa", "rc.web", split="validation")
    
    if num_samples:
        dataset = dataset.select(range(num_samples))
    
    validation_data = []
    for sample in dataset:
        search_results = sample["search_results"]
        docs = [
            {
                "title": search_results["title"][i],
                "url": search_results["url"][i],
                "description": search_results["description"][i],
                "filename": search_results["filename"][i],
                "rank": search_results["rank"][i],
                "search_context": search_results["search_context"][i],
            }
            for i in range(len(search_results["title"]))
        ]
        answer_aliases = sample["answer"]["aliases"]
        normalized_aliases = sample["answer"]["normalized_aliases"]
        all_answers = list(set(answer_aliases + normalized_aliases))
        
        validation_data.append({
            "query": sample["question"],
            "ground_truth": all_answers,
            "ground_truth_str": answer_aliases[0] if answer_aliases else "",
            "search_results": docs,
            "question_id": sample.get("question_id", None),
            "question_source": sample.get("question_source", None)
        })
    
    print(f"Loaded {len(validation_data)} validation queries")
    
    with open(cache_file, "wb") as f:
        pickle.dump(validation_data, f)
    
    return validation_data