import os
from typing import List
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
import evaluate
import shutil
from src.hipporag.utils.eval_utils import normalize_answer
# Load the BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load('meteor')

from src.hipporag import HippoRAG
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
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
    output_dir = 'outputs'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    # kinds_of_qa = [
    #     "closed_end",
    #     # "opened_end",  # Specify the type of QA task
    # ]  # Specify the type of QA task
    # majors = [
    #     "MCS",
    #     "DS",
    #     "AM"
    # ]  # Specify the majors
    embedding_model_names = [
        "text-embedding-3-small",
        "nvidia/NV-Embed-v2",
        "GritLM/GritLM-7B",
        "facebook/contriever",
    ]  # Specify the embedding model names
    dataset_styles = [
        "hcmus",  # Use the HCMUS dataset style
        None,  # Use the default dataset style
    ]
    # for major in majors:
    major = args.major
    kind_of_qa = args.kind_of_qa

    assert kind_of_qa in ["closed_end", "opened_end"], "kind_of_qa must be either 'closed_end' or 'opened_end'."
    # for kind_of_qa in kinds_of_qa:
    for dataset_style in dataset_styles:
        for embedding_model_name in embedding_model_names:
            cache_dir = os.path.join(
                output_dir,
                f"{major}_{kind_of_qa}_{dataset_style}_{embedding_model_name.split('/')[-1]}"
            )
            if os.path.exists(cache_dir):
                print(f"Cache directory {cache_dir} already exists. Cleaning...")
                # shutil.rmtree(cache_dir, ignore_errors=True)
                # os.makedirs(cache_dir, exist_ok=True)
            else:
                os.makedirs(cache_dir, exist_ok=True)
                corpus_path = f"data/rag_qa_test/{major}/{major}_corpus.json"
                with open(corpus_path, "r") as f:
                    corpus = json.load(f)

                docs = [f"Course ID: {doc['title']}\n{doc['text']}" for doc in corpus]

                # Startup a HippoRAG instance
                hipporag = HippoRAG(save_dir=cache_dir,
                                    llm_model_name=llm_model_name,
                                    embedding_model_name=embedding_model_name,
                                    dataset=dataset_style, ## HippoRAG base
                                    embedding_batch_size=2
                                    )

                # Run indexing
                hipporag.index(docs=docs)
                open_end_qa_ds = pd.DataFrame(json.load(open(f"data/rag_qa_test/{major}/{major}_{kind_of_qa}.json", "r")))
                queries = open_end_qa_ds["question"].tolist()
                raw_references = open_end_qa_ds["answer"].tolist()
                references = []
                for ref in raw_references:
                    if isinstance(ref, list):
                        references.append([normalize_answer(item) for item in ref])
                    else:
                        references.append([normalize_answer(ref)])
                gold_docs = [[f"Course ID: {item[0]['title']}\n{item[0]['text']}"] for item in open_end_qa_ds["paragraphs"].tolist()]
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
                predictions = [normalize_answer(pred) for pred in predictions]
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
                    "graph_info": hipporag.get_graph_info(),
                    "retrieval": overall_retrieval_result,
                    "open_ended": {
                        "bleu": open_end_results["bleu"],
                        "meteor": open_end_results["meteor"],
                        "rougeL": open_end_results["rougeL"],
                    },
                    "closed_ended": overall_qa_results,
                    "comparisons": comparisons,
                }
                ## save results
                save_fpath = os.path.join(
                    cache_dir,
                    "results.json"
                )
                with open(save_fpath, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Results saved to {save_fpath}")