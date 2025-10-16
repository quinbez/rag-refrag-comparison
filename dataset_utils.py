from datasets import load_dataset

def load_docs():
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

    return docs