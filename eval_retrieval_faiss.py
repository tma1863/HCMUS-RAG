from typing import List, Tuple, Dict, Any, Optional, Callable
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
from collections import Counter

from src.hipporag.utils.eval_utils import normalize_answer
# Load the BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load('meteor')

load_dotenv()

def calculate_retrieval_metric_scores(gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
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

def calculate_em_metric_scores(gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the Exact Match (EM) score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged EM score.
                - A list of dictionaries with EM scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results

def calculate_f1_metric_scores(gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the F1 score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged F1 score.
                - A list of dictionaries with F1 scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."


        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
            aggregated_f1 = aggregation_fn(f1_scores)
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results

if __name__ == "__main__":
    ## Reference: https://jayant017.medium.com/rag-using-langchain-part-3-vector-stores-and-retrievers-a75f4d14cbf3
    ## https://milvus.io/docs/integrate_with_langchain.md
    ## https://medium.com/@danushidk507/rag-vii-reranking-with-rrf-d8a13dba96de
    ## https://www.kaggle.com/code/raselmeya/exploring-rag-in-langchain-and-vector-bd
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

    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Use a smaller model for faster processing
    )
    # pca = PCA(n_components=3)
    llm = OpenAI(
        model="gpt-4o-mini"
    )

    chunk_size = 100  # Size of each chunk for the text splitter
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
            chunk.metadata["full_doc"] = doc_text
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

    found_docs = []
    predictions = []
    for query in tqdm(queries):
        query_embedding = openai_embeddings.embed_query(query)
        results = vector_store.similarity_search_by_vector(query_embedding, k=1000)
        full_doc_top1 = results[0].metadata["full_doc"]
        # combined_text = " ".join([result.page_content for result in results[:1]])
        # prompt = f"""
        # Based on the following text: `{full_doc_top1}`, answer the question: `{query}`. 
        # The answer should be concise and to the point. If the question is open-ended, provide a direct answer without additional explanations or context. If the question is closed-ended, respond with "Yes" or "No" as appropriate.
        # The output should be in format of "Answer: "
        # """
        prompt = f"""
        Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <context>
        {full_doc_top1}
        </context>

        <question>
        {query}
        </question>

        The response should be concise, to the point and use statistics or numbers when possible, while keeping the same writing style as in the given context information. If the question is open-ended, provide a direct answer without additional explanations or context. If the question is closed-ended, respond with "Yes" or "No" as appropriate.

        Assistant:
        """
        response = llm(prompt)
        response = response.split("Answer: ")[-1].strip().replace(".", "")
        predictions.append(normalize_answer(response))
        # predictions.append(
        #     # [result.page_content for result in results]
        #     [result.metadata["gold_id"] for result in results]
        # )
        tmp = []
        for result in results:
            gold_id = result.metadata["gold_id"]
            if gold_id not in tmp:
                tmp.append(gold_id)
        found_docs.append(tmp)
    pooled_retrieval_eval_results, _ = calculate_retrieval_metric_scores(gold_docs, found_docs, k_list=[1, 2, 5])
    pooled_em_eval_results, _ = calculate_em_metric_scores(references, predictions, aggregation_fn=np.max)
    pooled_f1_eval_results, _ = calculate_f1_metric_scores(references, predictions, aggregation_fn=np.max)
    pooled_qa_eval_results = pooled_em_eval_results
    pooled_qa_eval_results.update(pooled_f1_eval_results)
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)
    open_end_results = {
        "bleu": f"{bleu_results['bleu'] * 100:.2f}",
        "meteor": f"{meteor_results['meteor'] * 100:.2f}",
        "rougeL": f"{rouge_results['rougeL'] * 100:.2f}",
    }
    comparisons = [
        {
            "prediction": pred,
            "reference": ref,
        } for pred, ref in zip(predictions, references)
    ]
    results = {
        "retrieval": pooled_retrieval_eval_results,
        "closed_ended": pooled_qa_eval_results,
        "open_ended": {
            "bleu": open_end_results["bleu"],
            "meteor": open_end_results["meteor"],
            "rougeL": open_end_results["rougeL"],
        },
        "comparisons": comparisons,
    }
    save_fpath = os.path.join(
        cache_dir,
        "results.json"
    )
    with open(save_fpath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_fpath}")