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

from src.hipporag.StandardRAG import StandardRAG
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

from dotenv import load_dotenv
load_dotenv()
import gradio as gr

def chat_response(message, history):
    """
    Function to handle chat responses.
    """
    queries = [message]
    _, all_response_message, _ = hipporag.rag_qa(
        queries=queries
    )
    responses = [item.split("Answer: ")[1] for item in all_response_message]
    return responses[0]

def chat_response_streaming(message, history):
    """
    Function to handle chat responses with streaming.
    """
    queries = [message]
    _, all_response_message, _ = hipporag.rag_qa(
        queries=queries
    )
    responses = [item.split("Answer: ")[1] for item in all_response_message]
    # Simulate streaming by yielding one word at a time
    answer = responses[0]
    for i in range(1, len(answer) + 1):
        yield answer[:i]

if __name__ == "__main__":
    
    output_dir = 'outputs'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)    
    
    # for major in majors:
    embedding_model_name = "text-embedding-3-small"
    cache_dir = os.path.join(
        output_dir,
        f"demo_{embedding_model_name.split('/')[-1]}_standard_rag"
    )
    os.makedirs(cache_dir, exist_ok=True)
    docs = []
    for major in ["DS"]:
        corpus_path = f"data/rag_qa_test/{major}/{major}_corpus.json"
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        tmp = [f"Course ID: {doc['title']}\n{doc['text']}" for doc in corpus]
        docs.extend(tmp)
    llm_base_url = "https://api.openai.com/v1"
    llm_name = "gpt-4o-mini"  # Any OpenAI model name
    config = BaseConfig(
        save_dir=cache_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset="hcmus",  # Use the dataset style
        embedding_model_name=embedding_model_name,
        force_index_from_scratch=True,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=True,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=1,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode="online"
    )
    # Startup a HippoRAG instance
    hipporag = StandardRAG(global_config=config)

    # Run indexing
    hipporag.index(docs=docs)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Image(
                    "assets/hcmus_logo.png",
                    elem_id="logo",
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=60,
                    container=False,
                )
            with gr.Column(scale=6):
                gr.Markdown(
                    "<h1 style='text-align: center; margin-bottom: 0;'>Demo Hệ thống tư vấn môn học HCMUS</h1>",
                    elem_id="title"
                )
            with gr.Column(scale=2):
                gr.Image(
                    "assets/math_it_logo.png",
                    elem_id="logo",
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=60,
                    container=False,
                )
        gr.ChatInterface(
            chat_response_streaming, type="messages", autofocus=False
        )
    demo.launch()