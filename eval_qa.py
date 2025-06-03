import os
from typing import List
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
import evaluate
import shutil

# Load the BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load('meteor')

from src.hipporag import HippoRAG
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    cache_dir = 'outputs'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    save_dir = "results"
    llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    kinds_of_qa = [
        "closed_end",
        # "opened_end",  # Specify the type of QA task
    ]  # Specify the type of QA task
    majors = [
        "MCS",
        "DS",
        "AM"
    ]  # Specify the majors
    embedding_model_names = [
        "text-embedding-3-small",
        # "nvidia/NV-Embed-v2",
        # "GritLM/GritLM-7B",
        # "facebook/contriever",
    ]  # Specify the embedding model names
    dataset_styles = [
        "hcmus",  # Use the HCMUS dataset style
        None,  # Use the default dataset style
    ]
    for major in majors:
        for kind_of_qa in kinds_of_qa:
            for dataset_style in dataset_styles:
                for embedding_model_name in embedding_model_names:
                    shutil.rmtree(cache_dir, ignore_errors=True)

                    corpus_path = f"data/test/{major}/{major}_corpus.json"
                    with open(corpus_path, "r") as f:
                        corpus = json.load(f)

                    docs = [f"Course ID: {doc['title']}\n{doc['text']}" for doc in corpus]

                    # Startup a HippoRAG instance
                    hipporag = HippoRAG(save_dir=cache_dir,
                                        llm_model_name=llm_model_name,
                                        embedding_model_name=embedding_model_name,
                                        dataset=dataset_style, ## HippoRAG base
                                        embedding_batch_size=4
                                        )

                    # Run indexing
                    hipporag.index(docs=docs)
                    open_end_qa_ds = pd.DataFrame(json.load(open(f"data/test/{major}/{major}_{kind_of_qa}.json", "r")))
                    num_test = -1
                    queries = open_end_qa_ds["question"].tolist()[:num_test]
                    references = open_end_qa_ds["answer"].tolist()[:num_test]
                    gold_docs = [[f"Course ID: {item[0]['title']}\n{item[0]['text']}"] for item in open_end_qa_ds["paragraphs"].tolist()][:num_test]
                    # print(f"Query: {queries[0]}")
                    # queries_solutions, all_response_message, all_metadata = hipporag.rag_qa(queries=queries)
                    queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results = hipporag.rag_qa(
                        queries=queries,
                        gold_docs=gold_docs,
                        gold_answers=references
                    )
                    predictions = [item.split("Answer: ")[1] for item in all_response_message]
                    # breakpoint()
                    ## cleaning predictions and references
                    predictions = [pred.strip().replace(".", "").replace(r'"', "").lower() for pred in predictions]
                    references = [ref.strip().replace(".", "").replace(r'"', "").lower() for ref in references]
                    bleu_results = bleu_metric.compute(predictions=predictions, references=references)

                    rouge_results = rouge_metric.compute(predictions=predictions, references=references)

                    meteor_results = meteor_metric.compute(predictions=predictions, references=references)

                    open_end_results = {
                        "bleu": f"{bleu_results['bleu'] * 100:.2f}",
                        "meteor": f"{meteor_results['meteor'] * 100:.2f}",
                        "rougeL": f"{rouge_results['rougeL'] * 100:.2f}",
                    }
                    results = {
                        "major": major,
                        "dataset_style": dataset_style,
                        "embedding_model_name": embedding_model_name,
                        "graph_info": hipporag.get_graph_info(),
                        "retrieval_results": overall_retrieval_result,
                        "bleu": open_end_results["bleu"],
                        "meteor": open_end_results["meteor"],
                        "rougeL": open_end_results["rougeL"],
                        "overall_qa_results": overall_qa_results,
                        "predictions": predictions,
                        "references": references,
                    }
                    ## save results
                    results_save_path = os.path.join(save_dir, f"{major}_{kind_of_qa}_{dataset_style}_{embedding_model_name.replace('/', '_')}.json")
                    os.makedirs(save_dir, exist_ok=True)
                    with open(results_save_path, "w") as f:
                        json.dump(results, f, indent=4)
                    print(f"Results saved to {results_save_path}")