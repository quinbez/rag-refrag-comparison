import os
import pickle
from tqdm import tqdm
from datasets import load_dataset

# Cache directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _save_large_pickle(obj, path, chunk_size=20000):
    """Save large lists in chunks to prevent freezing."""
    temp_parts = []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = len(obj)

    print(f"Saving {total} items in chunks of {chunk_size}...")
    for i in range(0, total, chunk_size):
        chunk_path = f"{path}.part{i//chunk_size}"
        chunk = obj[i:i+chunk_size]

        with open(chunk_path, "wb") as f:
            pickle.dump(chunk, f)
        temp_parts.append(chunk_path)
        
        print(f"  Saved chunk {i//chunk_size + 1}/{(total + chunk_size - 1)//chunk_size}")

    # Save manifest
    manifest_path = path + ".manifest"
    with open(manifest_path, "wb") as f:
        pickle.dump(temp_parts, f)

    print(f"Dataset cached in {len(temp_parts)} chunks at {path}")

def _load_large_pickle(path):
    """Load large chunked pickle files."""
    manifest_path = path + ".manifest"

    if not os.path.exists(manifest_path):
        if os.path.exists(path):
            print(f"Loading from single pickle file: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Cache file not found: {path}")

    # Load chunked file
    print(f"Loading chunked pickle from {path}...")
    with open(manifest_path, "rb") as f:
        parts = pickle.load(f)

    data = []
    for i, part in enumerate(parts, 1):
        if not os.path.exists(part):
            raise FileNotFoundError(f"Missing chunk: {part}")

        with open(part, "rb") as f:
            chunk = pickle.load(f)
            data.extend(chunk)

        print(f"  Loaded chunk {i}/{len(parts)} ({len(chunk)} items)")
    print(f"Loaded {len(data)} total items")
    return data

def load_train_docs(force_reload=False, max_samples=None):
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
    if not force_reload and (os.path.exists(cache_file) or os.path.exists(cache_file + ".manifest")):
        try:
            print("Loading cached TriviaQA training docs...")
            docs = _load_large_pickle(cache_file)
            print(f"Loaded {len(docs)} documents from cache\n")
            return docs
        except Exception as e:
            print(f"Cache loading failed ({e}). Rebuilding...")
            for f in [cache_file, cache_file + ".manifest"]:
                if os.path.exists(f):
                    os.remove(f)

            cache_dir = os.path.dirname(cache_file)
            for f in os.listdir(cache_dir):
                if f.startswith(os.path.basename(cache_file) + ".part"):
                    os.remove(os.path.join(cache_dir, f))
    
    print("\n" + "="*70)
    print("Loading TriviaQA training set from HuggingFace (this may take a while)...")
    print("\n" + "="*70)

    dataset = load_dataset(
        "trivia_qa",
        "rc.web",
        split="train",
        cache_dir=os.path.join(CACHE_DIR, "huggingface")
    )
    print(f"Dataset loaded: {len(dataset)} samples\n")

    # Limit samples for testing
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Processing limited to {len(dataset)} samples for testing\n")

    print("Extracting documents from samples...")

    docs = []
    errors = 0
    for idx in tqdm(range(len(dataset)), desc="Processing"):
        try:
            sample = dataset[idx]
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
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n⚠ Error at sample {idx}: {e}")
            continue

    print(f"\nExtracted {len(docs)} documents (errors: {errors})")
    if len(docs) == 0:
        raise ValueError("No valid documents found for retrieval!")

    # Save to cache
    print(f"Caching dataset to {cache_file}...")
    _save_large_pickle(docs, cache_file, chunk_size=20000)

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
    
    if not force_reload and (os.path.exists(cache_file) or os.path.exists(cache_file + ".manifest")):
        try:
            print("Loading cached TriviaQA validation set...")
            validation_data = _load_large_pickle(cache_file)
            print(f"Loaded {len(validation_data)} validation queries\n")
            return validation_data
        except Exception as e:
            print(f"Cache loading failed ({e}). Rebuilding...")
            for f in [cache_file, cache_file + ".manifest"]:
                if os.path.exists(f):
                    os.remove(f)

    print("Loading TriviaQA validation set from HuggingFace...")
    dataset = load_dataset(
        "trivia_qa",
        "rc.web",
        split="validation",
        cache_dir=os.path.join(CACHE_DIR, "huggingface")
    )

    if num_samples:
        dataset = dataset.select(range(num_samples))
        print(f"Selected {num_samples} validation samples")

    validation_data = []
    errors = 0
    print("Processing validation samples...")
    for idx in tqdm(range(len(dataset)), desc="Validation"):
        try:
            sample = dataset[idx]
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
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError at validation sample {idx}: {e}")
            continue

    print(f"Loaded {len(validation_data)} validation queries (errors: {errors})")
    print(f"Caching validation set to {cache_file}...")
    _save_large_pickle(validation_data, cache_file, chunk_size=5000)

    return validation_data