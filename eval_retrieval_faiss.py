from typing import List, Tuple, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
# from sklearn.decomposition import PCA
import numpy as np
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import pandas as pd
from tqdm import tqdm
import evaluate
import argparse
import os
from src.hipporag.utils.eval_utils import normalize_answer
# Load the BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load('meteor')

load_dotenv()

def calculate_metric_scores(gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates Recall@k for each example and pools results for all queries.

    Args:
        gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
        retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
        k_list (List[int]): List of k values to calculate Recall@k for.

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A pooled dictionary with the averaged Recall@k across all examples.
            - A list of dictionaries with Recall@k for each example.
    """
    k_list = sorted(set(k_list))
    
    example_eval_results = []
    pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}
    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):        
        example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}

        # Compute Recall@k for each k
        for k in k_list:
            # Get top-k retrieved documents
            top_k_docs = example_retrieved_docs[:k]
            # Calculate intersection with gold documents
            relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
            # Compute recall
            if example_gold_docs:  # Avoid division by zero
                example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(set(example_gold_docs))
            else:
                example_eval_result[f"Recall@{k}"] = 0.0
        
        # Append example results
        example_eval_results.append(example_eval_result)
        
        # Accumulate pooled results
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

    # Average pooled results over all examples
    num_examples = len(gold_docs)
    for k in k_list:
        pooled_eval_results[f"Recall@{k}"] /= num_examples

    # round off to 4 decimal places for pooled results
    pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
    return pooled_eval_results, example_eval_results

if __name__ == "__main__":
    ## Reference: https://jayant017.medium.com/rag-using-langchain-part-3-vector-stores-and-retrievers-a75f4d14cbf3
    ## https://milvus.io/docs/integrate_with_langchain.md
    ## https://medium.com/@danushidk507/rag-vii-reranking-with-rrf-d8a13dba96de
    # Initialize OpenAI embeddings
    parser = argparse.ArgumentParser(description="Evaluate HippoRAG on QA tasks")
    parser.add_argument(
        "--major", type=str, default="MCS",
        help="Specify the major for evaluation (e.g., MCS, DS, AM)"
    )
    parser.add_argument(
        "--kind-of-qa", type=str, default="closed_end",
        help="Specify the kind of QA task (e.g., closed_end, opened_end)"
    )
    args = parser.parse_args()

    openai_embeddings = OpenAIEmbeddings()
    # pca = PCA(n_components=3)
    llm = OpenAI()

    chunk_size = 170  # Size of each chunk for the text splitter
    major = args.major
    kind_of_qa = args.kind_of_qa
    assert kind_of_qa in ["closed_end", "opened_end"], "kind_of_qa must be either 'closed_end' or 'opened_end'."
    output_dir = 'outputs'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    cache_dir = os.path.join(
        output_dir,
        f"{major}_{kind_of_qa}_faiss"
    )
    os.makedirs(cache_dir, exist_ok=True)
    corpus_path = f"data/rag_qa_test/{major}/{major}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"Course ID: {doc['title']}\n{doc['text']}" for doc in corpus]
    content = "; ".join(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    # Create a Document object with content
    # chunks = text_splitter.split_documents([Document(page_content=content)])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    all_chunks = []
    for doc in corpus:
        # Use doc['title'] or another field as the gold_id
        source_doc_id = doc['title']
        doc_text = f"Course ID: {source_doc_id}\n{doc['text']}"
        # Split this document into chunks (returns list of Document objects)
        chunks = text_splitter.create_documents([doc_text])
        # Add gold_id to each chunk's metadata
        for chunk in chunks:
            chunk.metadata["gold_id"] = source_doc_id
        all_chunks.extend(chunks)
    # texts = [chunk.page_content for chunk in chunks]
    # embeddings = openai_embeddings.embed_documents(texts)
    # Convert the list of lists to a NumPy array
    # embeddings_array = np.array(embeddings)
    # reduced_embeddings = pca.fit_transform(embeddings_array)
    # Create a list of tuples for texts and their embeddings
    # text_embeddings = list(zip(texts, embeddings))
    # Store the embeddings in a FAISS vector store using the pre-computed embeddings
    # vector_store = FAISS.from_embeddings(text_embeddings, openai_embeddings)
    vector_store = FAISS.from_documents(
        all_chunks, openai_embeddings,
    )
    open_end_qa_ds = pd.DataFrame(json.load(open(f"data/rag_qa_test/{major}/{major}_{kind_of_qa}.json", "r")))
    queries = open_end_qa_ds["question"].tolist()
    raw_references = open_end_qa_ds["answer"].tolist()
    references = []
    for ref in raw_references:
        if isinstance(ref, list):
            references.append([normalize_answer(item) for item in ref])
        else:
            references.append([normalize_answer(ref)])

    # gold_docs = [[f"Course ID: {item[0]['title']}\n{item[0]['text']}"] for item in open_end_qa_ds["paragraphs"].tolist()]
    gold_docs = [[f"{item[0]['title']}"] for item in open_end_qa_ds["paragraphs"].tolist()]

    predictions = []
    for query in queries:
        query_embedding = openai_embeddings.embed_query(query)
        results = vector_store.similarity_search_by_vector(query_embedding, k=10)
        # combined_text = " ".join([result.page_content for result in results])
        # prompt = f"""
        # Based on the following text, answer the question:\n\n{results[0]}\n\nQuestion: {query}.
        # You should present a concise, definitive response, devoid of additional elaborations, and keep the same writing style as the found source text if the question is a open-ended question else you can answer just Yes or No.
        # """
        # response = llm(prompt)
        # response = response.split("Answer: ")[-1].strip().replace(".", "")
        # predictions.append(normalize_answer(response))
        predictions.append(
            # [result.page_content for result in results]
            [result.metadata["gold_id"] for result in results]
        )
    pooled_eval_results, example_eval_results = calculate_metric_scores(gold_docs, predictions, k_list=[1, 2, 5])
    
    results = {
        "retrieval": pooled_eval_results
    }
    save_fpath = os.path.join(
        cache_dir,
        "results.json"
    )
    with open(save_fpath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_fpath}")