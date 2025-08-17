#!/usr/bin/env python3
"""
Main HippoRAG Implementation
===========================
Chức năng chính:
1. Index documents từ corpus
2. Chạy full evaluation trên test data
3. Generate comprehensive results

Usage:
    python main_hipporag.py --dataset AM --max_questions 10
    python main_hipporag.py --dataset DS --test_type closed_end
    python main_hipporag.py --dataset all --save_dir outputs/full_eval
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hipporag_complete import HippoRAG as HippoRAGComplete
from hipporag_evaluation import HippoRAGEvaluator
import json
import logging
import argparse
import time
import types
import difflib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Original rerank prompts
RERANK_SYSTEM_PROMPT = """You are a fact filtering expert. Your task is to identify the most relevant facts for answering a given question.

Guidelines:
1. Select facts that directly help answer the question
2. Include facts needed for multi-hop reasoning
3. Remove redundant or irrelevant facts
4. Maintain fact ordering by relevance
5. Return only the most important facts (max 4-5)"""

RERANK_USER_PROMPT = """Please filter these facts to answer the following question:

Question: {question}

Facts:
{facts}

Remember to:
- Keep only relevant facts
- Include facts for reasoning
- Remove redundancy
- Maintain relevance order
- Limit to 4-5 facts"""

def patch_rerank_with_detailed_logging(hipporag_instance):
    """
    Patch HippoRAG instance với detailed reranking logging
    Input: hipporag_instance - HippoRAG instance cần patch
    Output: None (patches instance in-place)
    Chức năng: Thêm chi tiết logging cho reranking process
    """
    
    def debug_rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """
        Debug version của rerank_facts với chi tiết logging
        Input: query (str), query_fact_scores (np.ndarray)
        Output: (fact_indices, facts, log_dict)
        Chức năng: Rerank facts với detailed logging
        """
        
        logger.info(f"RERANK DEBUG - Query: '{query}'")
        
        # Safe logging for potentially empty arrays
        if len(query_fact_scores) > 0:
            logger.info(f"Query fact scores: shape={query_fact_scores.shape}, min={query_fact_scores.min():.4f}, max={query_fact_scores.max():.4f}")
        else:
            logger.info(f"Query fact scores: shape={query_fact_scores.shape}, array is empty")
        
        # load args
        link_top_k: int = self.global_config.linking_top_k
        logger.info(f"Linking top_k setting: {link_top_k}")
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
                logger.info(f"Using all {len(query_fact_scores)} facts (fewer than top_k={link_top_k})")
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                logger.info(f"Selected top {link_top_k} facts from {len(query_fact_scores)} total")
                
            logger.info(f"Candidate indices: {candidate_fact_indices}")
            logger.info(f"Candidate scores: {[f'{query_fact_scores[i]:.4f}' for i in candidate_fact_indices]}")
            
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # Query to Triple (Top-5) - Facts trước rerank
            logger.info(f"\nQuery to Triple (Top-{min(5, len(candidate_facts))}):")
            logger.info("="*80)
            for i, fact in enumerate(candidate_facts[:5]):  # Chỉ hiển thị top 5
                score = query_fact_scores[candidate_fact_indices[i]]
                logger.info(f"  {i+1}. [Score: {score:.4f}] {fact}")
            if len(candidate_facts) > 5:
                logger.info(f"  ... and {len(candidate_facts) - 5} more facts")
            logger.info("="*80)
            
            # Format rerank prompt
            rerank_prompt = RERANK_USER_PROMPT.format(
                question=query,
                facts=json.dumps(candidate_facts, ensure_ascii=False)
            )
            
            # Rerank the facts
            logger.info(f"Starting LLM-based reranking...")
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            # Filtered Triple - Facts sau rerank
            logger.info(f"\nFiltered Triple ({len(top_k_facts)} facts):")
            logger.info("="*80)
            if len(top_k_facts) == 0:
                logger.warning("NO FACTS RETURNED AFTER RERANKING!")
            else:
                for i, fact in enumerate(top_k_facts):
                    logger.info(f"  {i+1}. {fact}")
            logger.info("="*80)
                
            logger.info(f"Reranker metadata: {reranker_dict}")
            
            # Compare before/after
            logger.info(f"\nRERANKING SUMMARY:")
            logger.info(f"  Query to Triple: {len(candidate_facts)} facts")
            logger.info(f"  Filtered Triple: {len(top_k_facts)} facts")
            logger.info(f"  Filtered out: {len(candidate_facts) - len(top_k_facts)} facts")
            logger.info(f"  Retention rate: {len(top_k_facts)/len(candidate_facts)*100:.1f}%")
            
            if len(top_k_facts) == 0:
                logger.warning("CRITICAL: Reranking returned ZERO facts! This will cause fallback to DPR.")
            
            rerank_log = {
                'query_to_triple_top5': candidate_facts[:5],
                'filtered_triple': top_k_facts,
                'facts_before_rerank': candidate_facts, 
                'facts_after_rerank': top_k_facts
            }
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}

    # Patch only the main rerank_facts method for detailed logging
    logger.info("Patching HippoRAG instance with detailed reranking logs...")
    
    # Patch main rerank_facts method
    hipporag_instance.rerank_facts = types.MethodType(debug_rerank_facts, hipporag_instance)
    
    logger.info("Reranking debug patches applied successfully!")

def load_corpus(dataset: str, base_dir: str = "QA for testing") -> List[str]:
    """
    Load corpus documents từ dataset
    Input: dataset (str), base_dir (str)
    Output: List[str] - list of document texts
    Chức năng: Load và extract text content từ corpus file
    """
    corpus_file = os.path.join(base_dir, dataset, f"{dataset}_corpus.json")
    
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        # Extract text content từ corpus
        documents = []
        for item in corpus_data:
            if 'text' in item:
                documents.append(item['text'])
                
        logger.info(f"Loaded {len(documents)} documents from {dataset} corpus")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading corpus for {dataset}: {e}")
        return []

def load_test_data(dataset: str, test_type: str, base_dir: str = "QA for testing") -> List[Dict]:
    """
    Load test questions từ dataset
    Input: dataset (str), test_type (str), base_dir (str)
    Output: List[Dict] - list of test questions
    Chức năng: Load test questions từ dataset file
    """
    test_file = os.path.join(base_dir, dataset, f"{dataset}_{test_type}.json")
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        logger.info(f"Loaded {len(test_data)} {test_type} questions from {dataset}")
        return test_data
        
    except Exception as e:
        logger.error(f"Error loading {test_type} questions for {dataset}: {e}")
        return []

def get_gold_docs(samples: List[Dict]) -> List[List[str]]:
    """Extract gold documents từ test samples theo format HippoRAG"""
    gold_docs = []
    
    for sample in samples:
        gold_doc = []
        
        if 'paragraphs' in sample:
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                # Combine title and text như HippoRAG gốc
                title = item.get('title', '')
                text = item.get('text', item.get('paragraph_text', ''))
                if title and text:
                    gold_doc.append(f"{title}\n{text}")
                elif text:
                    gold_doc.append(text)
        
        # Remove duplicates
        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    
    return gold_docs

def get_gold_answers(samples: List[Dict]) -> List[List[str]]:
    """Extract gold answers từ test samples theo format HippoRAG"""
    gold_answers = []
    
    for sample in samples:
        gold_ans = None
        
        if 'answer' in sample:
            gold_ans = sample['answer']
        elif 'gold_ans' in sample:
            gold_ans = sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        
        assert gold_ans is not None, f"No answer found in sample: {sample.get('id', 'unknown')}"
        
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        
        assert isinstance(gold_ans, list), f"Answer should be list, got {type(gold_ans)}"
        
        # Convert to set để remove duplicates, then back to list
        gold_ans = list(set(gold_ans))
        
        # Add answer aliases if available
        if 'answer_aliases' in sample:
            gold_ans.extend(sample['answer_aliases'])
            gold_ans = list(set(gold_ans))
        
        gold_answers.append(gold_ans)
    
    return gold_answers

def run_single_dataset_evaluation(dataset: str, test_type: str, save_dir: str, 
                                max_questions: Optional[int] = None,
                                force_index_from_scratch: bool = False,
                                debug_reranking: bool = False) -> Dict[str, Any]:
    """
    Run evaluation for a specific dataset and test type
    Input: dataset (str), test_type (str), save_dir (str), max_questions (int), force_index_from_scratch (bool), debug_reranking (bool)
    Output: Dict[str, Any] - evaluation results
    Chức năng: Share corpus indexing between test types
    """
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset.upper()} - {test_type}")
    print(f"{'='*60}")
    
    # Load corpus and questions
    evaluator = HippoRAGEvaluator(test_type=test_type)
    docs = load_corpus(dataset, "QA for testing")
    questions_data = load_test_data(dataset, test_type, "QA for testing")
    
    print(f"Loaded {len(docs)} documents")
    print(f"Loaded {len(questions_data)} questions")
    
    if max_questions and max_questions < len(questions_data):
        questions_data = questions_data[:max_questions]
        print(f"Limited to {len(questions_data)} questions")
    
    # SHARE CORPUS INDEXING: Use dataset_save_dir for corpus
    corpus_save_dir = os.path.join(save_dir, f"{dataset}_corpus_shared")
    
    # Handle force rebuild for corpus
    if force_index_from_scratch and os.path.exists(corpus_save_dir):
        import shutil
        print(f"Force rebuild: removing existing corpus data at {corpus_save_dir}")
        shutil.rmtree(corpus_save_dir)
    
    print(f"\nInitializing HippoRAG (corpus_save_dir: {corpus_save_dir})...")
    
    # Pass force_index_from_scratch to HippoRAG config
    try:
        from config.docker_config import docker_config as config
        config.force_index_from_scratch = force_index_from_scratch
    except ImportError:
        from config import Config
        config = Config()
        config.force_index_from_scratch = force_index_from_scratch
    
    hipporag = HippoRAGComplete(global_config=config, save_dir=corpus_save_dir)
    
    # Apply debug patches for detailed reranking logs
    if debug_reranking:
        print(f"Applying debug patches for reranking analysis...")
        patch_rerank_with_detailed_logging(hipporag)
    else:
        print(f"Debug reranking disabled (use --debug_reranking to enable)")
    
    # SMART CORPUS INDEXING: Incremental indexing approach
    corpus_indexed_file = os.path.join(corpus_save_dir, f"{dataset}_corpus_indexed.flag")
    
    # Initialize index_time variable
    index_time = 0
    needs_indexing = True
    
    # Check if corpus needs indexing
    if os.path.exists(corpus_indexed_file) and not force_index_from_scratch:
        try:
            with open(corpus_indexed_file, 'r') as f:
                flag_content = f.read().strip()
                # Extract previously indexed document count
                for line in flag_content.split('\n'):
                    if line.startswith('Documents:'):
                        prev_doc_count = int(line.split(':')[1].strip())
                        current_doc_count = len(docs)
                        
                        if prev_doc_count == current_doc_count:
                            print(f"Corpus {dataset} already indexed with {prev_doc_count} documents, reusing existing data")
                            print(f"Flag content: {flag_content}")
                            needs_indexing = False
                        else:
                            print(f"Corpus size changed: {prev_doc_count} → {current_doc_count} documents")
                            print(f"Performing incremental indexing for {current_doc_count - prev_doc_count} new documents...")
                            needs_indexing = True
                        break
                else:
                    # Flag file doesn't have document count, need to re-index
                    print(f"Flag file missing document count, performing full indexing...")
                    needs_indexing = True
        except Exception as e:
            print(f"Error reading flag file: {e}, performing full indexing...")
            needs_indexing = True
    else:
        if force_index_from_scratch:
            print(f"Force rebuild requested, performing full indexing...")
        else:
            print(f"No existing corpus found, performing initial indexing...")
        needs_indexing = True
    
    if needs_indexing:
        print(f"\nIndexing {len(docs)} documents for {dataset} corpus...")
        start_time = time.time()
        hipporag.index(docs)  # HippoRAG handles incremental automatically
        index_time = time.time() - start_time
        print(f"Corpus indexing completed in {index_time:.1f}s")
        
        # Update flag file with current document count
        os.makedirs(corpus_save_dir, exist_ok=True)
        with open(corpus_indexed_file, 'w') as f:
            f.write(f"Corpus {dataset} indexed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Documents: {len(docs)}\n")
            f.write(f"Index time: {index_time:.1f}s\n")
            f.write(f"Incremental indexing: {'Yes' if not force_index_from_scratch else 'No (forced rebuild)'}\n")
    
    # TEST-SPECIFIC RESULTS: Create separate dir for each test type to save results
    test_results_dir = os.path.join(save_dir, f"{dataset}_{test_type}_results")
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Run evaluation
    all_queries = [s['question'] for s in questions_data]
    gold_answers = get_gold_answers(questions_data)
    
    try:
        gold_docs = get_gold_docs(questions_data)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), \
            "Length of queries, gold_docs, and gold_answers should be the same."
        print(f"Extracted gold documents for retrieval evaluation")
    except Exception as e:
        logger.warning(f"Could not extract gold docs: {e}")
        gold_docs = None
    
    # Save test data to temporary file for evaluator
    temp_test_file = os.path.join(test_results_dir, "temp_test_data.json")
    os.makedirs(test_results_dir, exist_ok=True)
    
    with open(temp_test_file, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)
    
    # Start timing for evaluation
    eval_start_time = time.time()
    
    # Evaluate using HippoRAGEvaluator
    evaluator = HippoRAGEvaluator(include_advanced_metrics=True, test_type=test_type)
    detailed_results, aggregate_metrics = evaluator.evaluate_dataset(
        hipporag, 
        temp_test_file,
        include_retrieval=(gold_docs is not None),
        max_questions=max_questions
    )
    
    eval_time = time.time() - eval_start_time
    
    # Clean up temp file
    if os.path.exists(temp_test_file):
        os.remove(temp_test_file)
    
    print(f"Evaluation completed in {eval_time:.2f}s")
    
    # Print summary
    print(f"\nEVALUATION RESULTS:")
    print(f"  Questions: {len(detailed_results)}")
    print(f"  F1 Score: {aggregate_metrics.get('avg_f1_score', 0):.4f}")
    print(f"  Exact Match: {aggregate_metrics.get('avg_exact_match', 0):.4f}")
    if gold_docs:
        print(f"  Recall@1: {aggregate_metrics.get('avg_recall@1', 0):.4f}")
        print(f"  Recall@5: {aggregate_metrics.get('avg_recall@5', 0):.4f}")
    
    # Advanced metrics
    if 'avg_bleu3' in aggregate_metrics:
        print(f"  BLEU@3: {aggregate_metrics.get('avg_bleu3', 0):.4f}")
    if 'avg_bleu4' in aggregate_metrics:
        print(f"  BLEU@4: {aggregate_metrics.get('avg_bleu4', 0):.4f}")
    if 'avg_meteor' in aggregate_metrics:
        print(f"  METEOR: {aggregate_metrics.get('avg_meteor', 0):.4f}")
    if 'avg_rouge_l' in aggregate_metrics:
        print(f"  ROUGE-L: {aggregate_metrics.get('avg_rouge_l', 0):.4f}")
    
    # Prepare results
    results = {
        "dataset": dataset,
        "test_type": test_type,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_documents": len(docs),
        "num_questions": len(detailed_results),
        "index_time": index_time,
        "eval_time": eval_time,
        "system_stats": {
            "nodes": hipporag.graph.vcount(),
            "edges": hipporag.graph.ecount(),
            "chunks": hipporag.chunk_embedding_store.get_passage_count(),
            "entities": hipporag.entity_embedding_store.get_entity_count(),
            "facts": hipporag.fact_embedding_store.get_fact_count()
        },
        "aggregate_metrics": aggregate_metrics,
        "detailed_results": detailed_results
    }
    
    # Save results
    results_file = os.path.join(test_results_dir, f"{dataset}_{test_type}_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {results_file}")
    
    return results

def run_all_test_types_for_dataset(dataset: str, save_dir: str,
                                  max_questions: Optional[int] = None,
                                  force_index_from_scratch: bool = False,
                                  debug_reranking: bool = False,
                                  test_types: List[str] = None) -> Dict[str, Any]:
    """
    Run all test types for a dataset with shared corpus indexing
    Input: dataset (str), save_dir (str), max_questions (int), force_index_from_scratch (bool), debug_reranking (bool), test_types (List[str])
    Output: Dict[str, Any] - combined results from all test types
    Chức năng: Run multiple test types for a dataset with shared corpus indexing
    """
    
    if test_types is None:
        test_types = ['closed_end', 'opened_end', 'multihop', 'multihop2']
    
    print(f"\nRUNNING ALL TEST TYPES FOR DATASET: {dataset.upper()}")
    print(f"Test types: {', '.join(test_types)}")
    print(f"Shared corpus indexing: Enabled")
    print(f"{'='*70}")
    
    all_results = {}
    overall_start_time = time.time()
    
    for i, test_type in enumerate(test_types, 1):
        print(f"\n[{i}/{len(test_types)}] Processing {test_type}...")
        
        try:
            # Chỉ force rebuild corpus lần đầu tiên
            force_rebuild_this_round = force_index_from_scratch if i == 1 else False
            
            result = run_single_dataset_evaluation(
                dataset=dataset,
                test_type=test_type,
                save_dir=save_dir,
                max_questions=max_questions,
                force_index_from_scratch=force_rebuild_this_round,
                debug_reranking=debug_reranking
            )
            
            all_results[test_type] = result
            
        except Exception as e:
            print(f"Error in {test_type}: {str(e)}")
            all_results[test_type] = {'error': str(e)}
            logger.error(f"Error in {test_type}: {e}")
    
    overall_time = time.time() - overall_start_time
    
    # Create combined summary
    summary = {
        'dataset': dataset,
        'test_types': test_types,
        'total_time': overall_time,
        'shared_corpus': True,
        'individual_results': all_results
    }
    
    # Calculate average Results saved to
    valid_results = [r for r in all_results.values() if 'metrics' in r]
    if valid_results:
        avg_metrics = {}
        for metric in ['f1', 'exact_match', 'precision', 'recall']:
            scores = [r['metrics'].get(metric, 0) for r in valid_results]
            if scores:
                avg_metrics[f'avg_{metric}'] = sum(scores) / len(scores)
        
        summary['average_metrics'] = avg_metrics
        
        print(f"\nSUMMARY FOR {dataset.upper()}:")
        print(f"Total time: {overall_time:.1f}s")
        print(f"Average F1: {avg_metrics.get('avg_f1', 0):.4f}")
        print(f"Average EM: {avg_metrics.get('avg_exact_match', 0):.4f}")
    
    # Save combined results
    combined_results_file = os.path.join(save_dir, f"{dataset}_all_test_types_summary.json")
    with open(combined_results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Combined results saved to: {combined_results_file}")
    
    return summary

def run_interactive_mode(dataset: str, save_dir: str,
                        force_index_from_scratch: bool = False,
                        debug_reranking: bool = False) -> None:
    """
    Interactive mode - Load corpus and ask questions one by one
    Input: dataset (str), save_dir (str), force_index_from_scratch (bool), debug_reranking (bool)
    Output: None
    Chức năng: Interactive Q&A session with corpus indexing
    """
    
    print(f"\nINTERACTIVE MODE FOR DATASET: {dataset.upper()}")
    print(f"You can ask questions one by one after corpus indexing")
    print(f"{'='*70}")
    
    # Load corpus
    docs = load_corpus(dataset, "QA for testing")
    print(f"Loaded {len(docs)} documents from {dataset} corpus")
    
    # Setup corpus indexing directory
    corpus_save_dir = os.path.join(save_dir, f"{dataset}_interactive_corpus")
    
    # Handle force rebuild
    if force_index_from_scratch and os.path.exists(corpus_save_dir):
        import shutil
        print(f"Force rebuild: removing existing corpus data at {corpus_save_dir}")
        shutil.rmtree(corpus_save_dir)
    
    print(f"\nInitializing HippoRAG (corpus_save_dir: {corpus_save_dir})...")
    
    # Create config object for interactive mode (use docker_config in container)
    try:
        from config.docker_config import docker_config as config
        config.force_index_from_scratch = force_index_from_scratch
    except ImportError:
        from config import Config
        config = Config()
        config.force_index_from_scratch = force_index_from_scratch
    
    hipporag = HippoRAGComplete(global_config=config, save_dir=corpus_save_dir)
    
    # Apply debug patches if enabled
    if debug_reranking:
        print(f"Applying debug patches for reranking analysis...")
        patch_rerank_with_detailed_logging(hipporag)
    
    # Index corpus
    corpus_indexed_file = os.path.join(corpus_save_dir, f"{dataset}_corpus_indexed.flag")
    
    if not os.path.exists(corpus_indexed_file) or force_index_from_scratch:
        print(f"\nIndexing {len(docs)} documents for {dataset} corpus...")
        start_time = time.time()
        hipporag.index(docs)
        index_time = time.time() - start_time
        print(f"Corpus indexing completed in {index_time:.1f}s")
        
        # Create flag file
        os.makedirs(corpus_save_dir, exist_ok=True)
        with open(corpus_indexed_file, 'w') as f:
            f.write(f"Corpus {dataset} indexed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Documents: {len(docs)}\n")
            f.write(f"Index time: {index_time:.1f}s\n")
    else:
        print(f"Corpus {dataset} already indexed, reusing existing data")
        with open(corpus_indexed_file, 'r') as f:
            flag_content = f.read().strip()
            print(f"Flag content: {flag_content}")
    
    # Show graph statistics and check for rebuild need
    try:
        graph_info = hipporag.get_graph_info()
        print(f"\nKnowledge Graph Statistics:")
        print("="*50)
        for key, value in graph_info.items():
            print(f"  {key}: {value:,}")
        print("="*50)
        
        # Check if graph needs rebuilding from existing data
        has_embedding_data = (
            graph_info.get('num_phrase_nodes', 0) > 0 or 
            graph_info.get('num_passage_nodes', 0) > 0 or 
            graph_info.get('num_extracted_triples', 0) > 0
        )
        graph_is_empty = graph_info.get('unique_nodes_N', 0) == 0
        
        if has_embedding_data and graph_is_empty and os.path.exists(hipporag.openie_results_path):
            print(f"\nDetected embedding data but empty graph. Rebuilding graph from existing OpenIE results...")
            try:
                # Load existing OpenIE results
                with open(hipporag.openie_results_path, 'r', encoding='utf-8') as f:
                    openie_data = json.load(f)
                
                openie_docs = openie_data.get('docs', [])
                if len(openie_docs) > 0:
                    # Trigger graph construction from existing data
                    start_time = time.time()
                    
                    # Prepare data like in index() method
                    from utils.hipporag_utils import reformat_openie_results, extract_entity_nodes, flatten_facts, text_processing
                    ner_results_dict, triple_results_dict = reformat_openie_results(openie_docs)
                    
                    chunk_ids = [doc['idx'] for doc in openie_docs]
                    chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
                    entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
                    facts = flatten_facts(chunk_triples)
                    
                    # Construct graph
                    hipporag.node_to_node_stats = {}
                    hipporag.ent_node_to_chunk_ids = {}
                    
                    hipporag.add_fact_edges(chunk_ids, chunk_triples)
                    num_new_chunks = hipporag.add_passage_edges(chunk_ids, chunk_triple_entities)
                    
                    if len(hipporag.node_to_node_stats) > 0 or len(hipporag.ent_node_to_chunk_ids) > 0:
                        hipporag.add_synonymy_edges()
                        hipporag.augment_graph()
                        hipporag.save_igraph()
                        
                        rebuild_time = time.time() - start_time
                        print(f"Graph rebuilt in {rebuild_time:.1f}s!")
                        
                        # Show updated stats
                        updated_graph_info = hipporag.get_graph_info()
                        print(f"\nUpdated Graph Statistics:")
                        print("="*50)  
                        for key, value in updated_graph_info.items():
                            print(f"  {key}: {value:,}")
                        print("="*50)
                    else:
                        print("No relationships found to build graph")
                        
            except Exception as e:
                print(f"Error rebuilding graph: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Could not get graph info: {e}")
    
    # Start interactive session
    print(f"\nInteractive Q&A Session for {dataset.upper()}")
    print("="*60)
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - 'context on/off' to toggle context display")
    print("  - 'debug on/off' to toggle debug mode") 
    print("  - 'stats' to show graph statistics")
    print("  - 'save <filename>' to save session")
    print("  - 'quit' or 'exit' to end session")
    print("="*60)
    
    show_context = True
    session_history = []
    question_count = 0
    
    while True:
        try:
            # Safer input with EOF handling and encoding fix
            try:
                question = input(f"\nQuestion #{question_count + 1}: ").strip()
                
                # Handle character encoding issues
                try:
                    # Try to encode/decode to fix any encoding issues
                    question = question.encode('utf-8', errors='ignore').decode('utf-8')
                    # Remove any problematic Unicode characters
                    question = ''.join(char for char in question if ord(char) < 65536)
                    # Clean up extra spaces
                    question = ' '.join(question.split())
                    
                    if question:
                        logger.info(f"Cleaned question: '{question}'")
                    
                except Exception as encoding_error:
                    logger.warning(f"Encoding issue detected: {encoding_error}")
                    print(f"Your input has encoding issues. Please try typing in English.")
                    continue
                    
            except EOFError:
                print("\nEOF detected - probably running in non-interactive environment")
                print("Use: docker exec -it <container_name> python main_hipporag.py --dataset DS --interactive")
                break
            except KeyboardInterrupt:
                print("\n\nInteractive session interrupted!")
                break
                
            if question.lower() in ['quit', 'exit']:
                print("Interactive session ended!")
                break
                
            if question.lower() == 'context on':
                show_context = True
                print("Context display enabled")
                continue
                
            if question.lower() == 'context off':
                show_context = False
                print("Context display disabled")
                continue
                
            if question.lower() == 'debug on':
                if not debug_reranking:
                    patch_rerank_with_detailed_logging(hipporag)
                    debug_reranking = True
                print("Debug mode enabled")
                continue
                
            if question.lower() == 'debug off':
                print("Debug mode disabled (restart needed to fully disable)")
                continue
                
            if question.lower() == 'stats':
                try:
                    graph_info = hipporag.get_graph_info()
                    print(f"\nCurrent Graph Statistics:")
                    print("="*40)
                    for key, value in graph_info.items():
                        print(f"  {key}: {value:,}")
                    print("="*40)
                except Exception as e:
                    print(f"Error getting stats: {e}")
                continue
                
            if question.lower().startswith('save '):
                filename = question[5:].strip()
                if not filename:
                    filename = f"{dataset}_interactive_session_{time.strftime('%Y%m%d_%H%M%S')}.json"
                
                try:
                    session_file = os.path.join(save_dir, filename)
                    session_data = {
                        'dataset': dataset,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'total_questions': len(session_history),
                        'session_history': session_history
                    }
                    
                    with open(session_file, 'w', encoding='utf-8') as f:
                        json.dump(session_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"Session saved to: {session_file}")
                except Exception as e:
                    print(f"Error saving session: {e}")
                continue
            
            if not question:
                print("Empty question. Please type a question or 'quit' to exit.")
                continue
            
            # Check for graph empty and suggest rebuild
            if hasattr(hipporag, 'graph') and hipporag.graph is not None and hipporag.graph.vcount() == 0:
                print("Knowledge graph is empty. This will cause poor results.")
                print("Consider running with --force_index_from_scratch to rebuild the corpus.")
                rebuild = input("Would you like me to rebuild the corpus now? (y/N): ").strip().lower()
                if rebuild in ['y', 'yes']:
                    print("Rebuilding corpus... This may take several minutes.")
                    try:
                        docs = load_corpus(dataset, "QA for testing")
                        start_time = time.time()
                        # Set force rebuild flag
                        hipporag.global_config.force_index_from_scratch = True
                        hipporag.index(docs)
                        rebuild_time = time.time() - start_time
                        print(f"Corpus rebuilt in {rebuild_time:.1f}s!")
                        
                        # Update flag file
                        corpus_indexed_file = os.path.join(hipporag.save_dir, f"{dataset}_corpus_indexed.flag")
                        with open(corpus_indexed_file, 'w') as f:
                            f.write(f"Corpus {dataset} rebuilt at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Documents: {len(docs)}\n")
                            f.write(f"Rebuild time: {rebuild_time:.1f}s\n")
                            f.write("Reason: Graph was empty in interactive mode\n")
                        
                        # Show new stats
                        try:
                            graph_info = hipporag.get_graph_info()
                            print(f"\nNew Graph Statistics:")
                            print("="*40)
                            for key, value in graph_info.items():
                                print(f"  {key}: {value:,}")
                            print("="*40)
                        except Exception as e:
                            print(f"Could not get new graph stats: {e}")
                            
                    except Exception as e:
                        print(f"Error rebuilding corpus: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            print(f"\nProcessing question: {question}")
            start_time = time.time()
            
            try:
                # Ensure graph is initialized before processing
                if hipporag.graph is None:
                    logger.info("Graph not loaded, initializing...")
                    hipporag.graph = hipporag.initialize_graph()
                    logger.info("Graph initialized successfully")
                
                # Use rag_qa for single question
                query_solutions, all_responses, all_metadata = hipporag.rag_qa([question])
                
                query_solution = query_solutions[0]
                answer = query_solution.answer
                retrieved_docs = query_solution.docs[:5]  # Top 5 docs
                doc_scores = query_solution.doc_scores[:5]
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Display results
                print("\n" + "="*80)
                print(f"Answer: {answer}")
                print(f"Processing time: {processing_time:.2f} seconds")
                
                if show_context and retrieved_docs:
                    print(f"\nRetrieved Context (Top {len(retrieved_docs)} documents):")
                    for i, (doc, score) in enumerate(zip(retrieved_docs, doc_scores), 1):
                        print(f"\n[{i}] Score: {score:.4f}")
                        print(f"    {doc[:300]}{'...' if len(doc) > 300 else ''}")
                
                print("="*80)
                
                # Save to session history
                question_count += 1
                session_record = {
                    'question_number': question_count,
                    'question': question,
                    'answer': answer,
                    'processing_time': processing_time,
                    'retrieved_docs_count': len(retrieved_docs),
                    'top_doc_score': doc_scores[0] if doc_scores else 0,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                session_history.append(session_record)
                
            except Exception as e:
                print(f"Error processing question: {e}")
                if debug_reranking:
                    import traceback
                    traceback.print_exc()
                
        except KeyboardInterrupt:
            print("\n\nInteractive session interrupted!")
            break
        except Exception as e:
            print(f"Session error: {e}")
            logger.error(f"Session error details: {e}")
            import traceback
            traceback.print_exc()
            print("TIP: Make sure you're running with 'docker exec -it' for interactive mode")
            print("If you see encoding errors, try typing questions in simple English")
    
    # Session summary
    if session_history:
        print(f"\nSESSION SUMMARY:")
        print(f"  Total questions asked: {len(session_history)}")
        avg_time = sum(q['processing_time'] for q in session_history) / len(session_history)
        print(f"  Average processing time: {avg_time:.2f}s")
        
        # Auto-save session
        auto_save_file = os.path.join(save_dir, f"{dataset}_interactive_session_auto.json")
        try:
            session_data = {
                'dataset': dataset,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_questions': len(session_history),
                'session_history': session_history
            }
            
            with open(auto_save_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"Session auto-saved to: {auto_save_file}")
        except Exception as e:
            print(f"Could not auto-save session: {e}")

def main():
    """
    Main function with argument parsing
    Input: command line arguments
    Output: None
    Function: Parse arguments and run appropriate evaluation mode
    """
    parser = argparse.ArgumentParser(description='HippoRAG Complete Evaluation')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['AM', 'DS', 'MCS', 'all'],
                       help='Dataset to evaluate (AM/DS/MCS/all)')
    parser.add_argument('--test_type', type=str, default='all',
                       choices=['closed_end', 'opened_end', 'multihop', 'multihop2', 'all'],
                       help='Type of test to run (default: all)')
    parser.add_argument('--save_dir', type=str, default='outputs/main_evaluation',
                       help='Directory to save results (default: outputs/main_evaluation)')
    parser.add_argument('--max_questions', type=int, default=None,
                       help='Maximum number of questions to process (default: all)')
    parser.add_argument('--force_index_from_scratch', action='store_true',
                       help='Force reindexing from scratch')
    parser.add_argument('--base_dir', type=str, default='QA for testing',
                       help='Base directory for QA testing data (default: QA for testing)')
    parser.add_argument('--debug_reranking', action='store_true',
                       help='Enable detailed reranking debug logs')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode - ask questions one by one')
    
    args = parser.parse_args()
    
    # Print configuration
    print("HippoRAG Complete Implementation - Main Evaluation")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    if not args.interactive:
        print(f"Test Type: {args.test_type}")
        print(f"Max Questions: {args.max_questions or 'All'}")
    else:
        print(f"Mode: Interactive Q&A")
    print(f"Save Dir: {args.save_dir}")
    print(f"Force Rebuild: {args.force_index_from_scratch}")
    print(f"Debug Reranking: {args.debug_reranking}")
    print("="*60)
    
    try:
        # Check for interactive mode first
        if args.interactive:
            if args.dataset == 'all':
                print("Interactive mode not supported with 'all' datasets")
                print("Please specify a single dataset: AM, DS, or MCS")
                return
            
            print(f"Starting interactive mode for dataset: {args.dataset}")
            run_interactive_mode(
                dataset=args.dataset,
                save_dir=args.save_dir,
                force_index_from_scratch=args.force_index_from_scratch,
                debug_reranking=args.debug_reranking
            )
            return
        
        # Existing evaluation logic
        if args.dataset == 'all':
            # Run all datasets with all test types
            print("Running evaluation for ALL DATASETS")
            all_results = {}
            
            for dataset in ['AM', 'DS', 'MCS']:
                print(f"\nProcessing dataset: {dataset}")
                try:
                    result = run_all_test_types_for_dataset(
                        dataset=dataset,
                        save_dir=args.save_dir,
                        max_questions=args.max_questions,
                        force_index_from_scratch=args.force_index_from_scratch,
                        debug_reranking=args.debug_reranking
                    )
                    all_results[dataset] = result
                except Exception as e:
                    print(f"Error processing dataset {dataset}: {e}")
                    all_results[dataset] = {'error': str(e)}
            
            # Save combined results for all datasets
            all_datasets_file = os.path.join(args.save_dir, "all_datasets_summary.json")
            with open(all_datasets_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"\nAll datasets evaluation completed!")
            print(f"Results saved to: {all_datasets_file}")
            
        else:
            # Single dataset
            if args.test_type == 'all':
                # Run all test types for the dataset with shared corpus
                print(f"Running ALL test types for dataset: {args.dataset}")
                results = run_all_test_types_for_dataset(
                    dataset=args.dataset,
                    save_dir=args.save_dir,
                    max_questions=args.max_questions,
                    force_index_from_scratch=args.force_index_from_scratch,
                    debug_reranking=args.debug_reranking
                )
                
                print(f"\nAll test types completed for {args.dataset}!")
                
            else:
                # Run single test type
                print(f"Running {args.test_type} for dataset: {args.dataset}")
                results = run_single_dataset_evaluation(
                    dataset=args.dataset,
                    test_type=args.test_type,
                    save_dir=args.save_dir,
                    max_questions=args.max_questions,
                    force_index_from_scratch=args.force_index_from_scratch,
                    debug_reranking=args.debug_reranking
                )
                
                if results and 'metrics' in results:
                    metrics = results['metrics']
                    print(f"\nEvaluation completed!")
                    print(f"F1 Score: {metrics.get('f1', 0):.4f}")
                    print(f"Exact Match: {metrics.get('exact_match', 0):.4f}")
                    print(f"Precision: {metrics.get('precision', 0):.4f}")
                    print(f"Recall: {metrics.get('recall', 0):.4f}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        logger.error(f"Main evaluation error: {e}")
        raise

if __name__ == "__main__":
    main() 