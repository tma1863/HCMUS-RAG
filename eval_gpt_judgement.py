import os
from typing import List
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
import evaluate
import shutil
from langchain.llms import OpenAI

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
    llm = OpenAI()
    embedding_model_names = [
        "text-embedding-3-small",
        "nvidia/NV-Embed-v2",
        "GritLM/GritLM-7B",
        "facebook/contriever",
    ]  # Specify the embedding model names
    # for major in majors:
    major = args.major
    kind_of_qa = args.kind_of_qa

    assert kind_of_qa in ["closed_end", "opened_end"], "kind_of_qa must be either 'closed_end' or 'opened_end'."
    # for kind_of_qa in kinds_of_qa:
    for embedding_model_name in embedding_model_names:
        cache_dir = os.path.join(
            output_dir,
            f"{major}_{kind_of_qa}_hcmus_{embedding_model_name.split('/')[-1]}"
        )
        results_fpath = os.path.join(
            cache_dir,
            "results.json"
        )
        comparisons = json.load(open(results_fpath, "r"))["comparisons"]
        open_end_qa_ds = pd.DataFrame(json.load(open(f"data/rag_qa_test/{major}/{major}_{kind_of_qa}.json", "r")))
        queries = open_end_qa_ds["question"].tolist()
        gpt_judgement_results = []
        for i in tqdm(range(len(queries))):
            query = queries[i]
            pred_answer = comparisons[i]["prediction"]
            gt_answer = comparisons[i]["reference"][0]
            prompt = f"""You are a judge who evaluates the quality of answers to questions. You should provide a score in the scale of 1 to 10 (without explanation), where 1 is the predicted answer is completely irrelevant to the ground truth answer or the question, and 10 is the predicted answer is exactly the same as the ground truth answer in the semantic meaning.
            The question is: {query}. The predicted answer is: {pred_answer}. The ground truth answer is: {gt_answer}.
            """
            response = llm(prompt)
            gpt_judgement_results.append(response.strip())
        results = json.load(open(results_fpath, "r"))
        results["gpt_judgement"] = gpt_judgement_results
        with open(results_fpath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_fpath}")