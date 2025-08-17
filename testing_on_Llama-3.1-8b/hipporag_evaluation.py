#!/usr/bin/env python3
"""
HippoRAG Compatible Evaluation System
------------------------------------
Hệ thống đánh giá tương thích với HippoRAG, implement các metric:
- QA Exact Match
- QA F1 Score  
- Retrieval Recall@k
- BLEU@4
- METEOR
- ROUGE-L
"""

import json
import re
import string
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from collections import Counter
import logging

# Import for advanced metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import nltk
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
    ADVANCED_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced metrics not available. Install with: pip install nltk rouge-score")
    print(f"Error: {e}")
    ADVANCED_METRICS_AVAILABLE = False

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_answer(answer: str) -> str:
    """
    Normalize answer theo chính xác chuẩn HippoRAG gốc + clean quotes.
    
    Args:
        answer: Câu trả lời cần chuẩn hóa
        
    Returns:
        Câu trả lời đã được chuẩn hóa
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def clean_quotes(text):
        # Remove escaped quotes and regular quotes
        return text.replace('\\"', '').replace('"', '').replace("'", "")
    
    # Add clean_quotes step to the original flow
    return white_space_fix(remove_articles(remove_punc(lower(clean_quotes(answer)))))

@dataclass
class QuerySolution:
    """Query solution tương thích với HippoRAG"""
    question: str
    docs: List[str]
    doc_scores: Optional[List[float]] = None
    answer: Optional[str] = None
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores[:5]] if self.doc_scores else None,
            "gold_docs": self.gold_docs,
        }

class BaseMetric:
    """Base metric class theo chuẩn HippoRAG"""
    
    def __init__(self):
        self.metric_name = "base"
        
    def calculate_metric_scores(self) -> Tuple[Dict[str, Union[int, float]], List[Union[int, float]]]:
        """Calculate metric scores"""
        return {}, []

class QAExactMatch(BaseMetric):
    """QA Exact Match metric theo chuẩn HippoRAG"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "qa_exact_match"

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str], 
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate Exact Match scores.
        
        Args:
            gold_answers: List of lists containing ground truth answers
            predicted_answers: List of predicted answers
            aggregation_fn: Function to aggregate scores across multiple gold answers
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 
                        for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results

class QAF1Score(BaseMetric):
    """QA F1 Score metric theo chuẩn HippoRAG"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "qa_f1_score"

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str], 
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate F1 scores.
        
        Args:
            gold_answers: List of lists containing ground truth answers
            predicted_answers: List of predicted answers
            aggregation_fn: Function to aggregate scores across multiple gold answers
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens) if predicted_tokens else 0.0
            recall = 1.0 * num_same / len(gold_tokens) if gold_tokens else 0.0
            
            if precision + recall == 0:
                return 0.0
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

class RetrievalRecall(BaseMetric):
    """Retrieval Recall metric theo chuẩn HippoRAG"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "retrieval_recall"
        
    def calculate_metric_scores(self, 
                              gold_docs: List[List[str]], 
                              retrieved_docs: List[List[str]], 
                              k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate Recall@k for each example với exact matching như HippoRAG gốc.
        
        Args:
            gold_docs: List of lists containing ground truth documents
            retrieved_docs: List of lists containing retrieved documents
            k_list: List of k values to calculate Recall@k for
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        k_list = sorted(set(k_list))
        
        example_eval_results = []
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}
        
        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
            if len(example_retrieved_docs) < k_list[-1]:
                logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) "
                             f"is smaller than largest topk for recall score ({k_list[-1]})")
            
            example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}
  
            # Compute Recall@k for each k
            for k in k_list:
                # Get top-k retrieved documents
                top_k_docs = example_retrieved_docs[:k]
                
                # Calculate intersection with gold documents (exact matching như HippoRAG gốc)
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

        # Round off to 4 decimal places for pooled results
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results

class BLEU4Score(BaseMetric):
    """BLEU@4 Score metric for answer quality evaluation"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "bleu4_score"
        self.smoothing_function = SmoothingFunction().method1 if ADVANCED_METRICS_AVAILABLE else None

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str], 
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate BLEU@4 scores.
        
        Args:
            gold_answers: List of lists containing ground truth answers
            predicted_answers: List of predicted answers
            aggregation_fn: Function to aggregate scores across multiple gold answers
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        if not ADVANCED_METRICS_AVAILABLE:
            logger.warning("BLEU@4 metric not available. Install nltk and rouge-score.")
            return {"BLEU@4": 0.0}, [{"BLEU@4": 0.0} for _ in predicted_answers]
            
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."

        def compute_bleu4(gold: str, predicted: str) -> float:
            # Tokenize
            gold_tokens = gold.lower().split()
            predicted_tokens = predicted.lower().split()
            
            if not predicted_tokens:
                return 0.0
                
            # BLEU expects list of reference sentences
            references = [gold_tokens]
            
            try:
                score = sentence_bleu(references, predicted_tokens, 
                                    weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=self.smoothing_function)
                return score
            except:
                return 0.0

        example_eval_results = []
        total_bleu4 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            bleu4_scores = [compute_bleu4(gold, predicted) for gold in gold_list]
            aggregated_bleu4 = aggregation_fn(bleu4_scores)
            example_eval_results.append({"BLEU@4": aggregated_bleu4})
            total_bleu4 += aggregated_bleu4

        avg_bleu4 = total_bleu4 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"BLEU@4": avg_bleu4}

        return pooled_eval_results, example_eval_results

class BLEU3Score(BaseMetric):
    """BLEU@3 Score metric for answer quality evaluation (for closed-end questions)"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "bleu3_score"
        self.smoothing_function = SmoothingFunction().method1 if ADVANCED_METRICS_AVAILABLE else None

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str], 
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate BLEU@3 scores.
        
        Args:
            gold_answers: List of lists containing ground truth answers
            predicted_answers: List of predicted answers
            aggregation_fn: Function to aggregate scores across multiple gold answers
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        if not ADVANCED_METRICS_AVAILABLE:
            logger.warning("BLEU@3 metric not available. Install nltk and rouge-score.")
            return {"BLEU@3": 0.0}, [{"BLEU@3": 0.0} for _ in predicted_answers]
            
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."

        def compute_bleu3(gold: str, predicted: str) -> float:
            # Tokenize
            gold_tokens = gold.lower().split()
            predicted_tokens = predicted.lower().split()
            
            if not predicted_tokens:
                return 0.0
                
            # BLEU expects list of reference sentences
            references = [gold_tokens]
            
            try:
                # BLEU@3 uses weights for 1-gram, 2-gram, 3-gram only
                score = sentence_bleu(references, predicted_tokens, 
                                    weights=(0.333, 0.333, 0.333, 0),
                                    smoothing_function=self.smoothing_function)
                return score
            except:
                return 0.0

        example_eval_results = []
        total_bleu3 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            bleu3_scores = [compute_bleu3(gold, predicted) for gold in gold_list]
            aggregated_bleu3 = aggregation_fn(bleu3_scores)
            example_eval_results.append({"BLEU@3": aggregated_bleu3})
            total_bleu3 += aggregated_bleu3

        avg_bleu3 = total_bleu3 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"BLEU@3": avg_bleu3}

        return pooled_eval_results, example_eval_results

class METEORScore(BaseMetric):
    """METEOR Score metric for answer quality evaluation"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "meteor_score"

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str], 
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate METEOR scores.
        
        Args:
            gold_answers: List of lists containing ground truth answers
            predicted_answers: List of predicted answers
            aggregation_fn: Function to aggregate scores across multiple gold answers
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        if not ADVANCED_METRICS_AVAILABLE:
            logger.warning("METEOR metric not available. Install nltk and rouge-score.")
            return {"METEOR": 0.0}, [{"METEOR": 0.0} for _ in predicted_answers]
            
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."

        def compute_meteor(gold: str, predicted: str) -> float:
            try:
                # METEOR expects tokenized strings
                gold_tokens = gold.lower().split()
                predicted_tokens = predicted.lower().split()
                
                if not predicted_tokens or not gold_tokens:
                    return 0.0
                
                score = meteor_score([gold_tokens], predicted_tokens)
                return score
            except:
                return 0.0

        example_eval_results = []
        total_meteor = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            meteor_scores = [compute_meteor(gold, predicted) for gold in gold_list]
            aggregated_meteor = aggregation_fn(meteor_scores)
            example_eval_results.append({"METEOR": aggregated_meteor})
            total_meteor += aggregated_meteor

        avg_meteor = total_meteor / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"METEOR": avg_meteor}

        return pooled_eval_results, example_eval_results

class ROUGELScore(BaseMetric):
    """ROUGE-L Score metric for answer quality evaluation"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "rouge_l_score"
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if ADVANCED_METRICS_AVAILABLE else None

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str], 
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate ROUGE-L scores.
        
        Args:
            gold_answers: List of lists containing ground truth answers
            predicted_answers: List of predicted answers
            aggregation_fn: Function to aggregate scores across multiple gold answers
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        if not ADVANCED_METRICS_AVAILABLE:
            logger.warning("ROUGE-L metric not available. Install nltk and rouge-score.")
            return {"ROUGE-L": 0.0}, [{"ROUGE-L": 0.0} for _ in predicted_answers]
            
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."

        def compute_rouge_l(gold: str, predicted: str) -> float:
            try:
                scores = self.scorer.score(gold, predicted)
                # Return F1 score for ROUGE-L
                return scores['rougeL'].fmeasure
            except:
                return 0.0

        example_eval_results = []
        total_rouge_l = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            rouge_l_scores = [compute_rouge_l(gold, predicted) for gold in gold_list]
            aggregated_rouge_l = aggregation_fn(rouge_l_scores)
            example_eval_results.append({"ROUGE-L": aggregated_rouge_l})
            total_rouge_l += aggregated_rouge_l

        avg_rouge_l = total_rouge_l / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ROUGE-L": avg_rouge_l}

        return pooled_eval_results, example_eval_results

class SuccessRate(BaseMetric):
    """Success Rate metric for multi-round evaluation"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "success_rate"

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str], 
                              num_rounds: int = 5,
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate Success Rate over multiple rounds.
        
        Args:
            gold_answers: List of lists containing ground truth answers
            predicted_answers: List of predicted answers
            num_rounds: Number of rounds to evaluate (default: 5)
            aggregation_fn: Function to aggregate scores across multiple gold answers
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_success = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            # For each example, check if any gold answer matches predicted
            success_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 
                            for gold in gold_list]
            is_success = aggregation_fn(success_scores) > 0
            
            # Simulate multi-round success (for now, use single round result)
            # In real implementation, this would involve multiple attempts
            round_success = is_success
            
            example_eval_results.append({
                f"SuccessRate_{num_rounds}rounds": 1.0 if round_success else 0.0
            })
            total_success += (1.0 if round_success else 0.0)

        avg_success = total_success / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {f"SuccessRate_{num_rounds}rounds": avg_success}

        return pooled_eval_results, example_eval_results

class TaskSuccess(BaseMetric):
    """Task Success metric for evaluating task completion"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "task_success"

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str], 
                              task_criteria: Optional[Dict] = None,
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate Task Success based on task completion criteria.
        
        Args:
            gold_answers: List of lists containing ground truth answers
            predicted_answers: List of predicted answers
            task_criteria: Optional criteria for task success evaluation
            aggregation_fn: Function to aggregate scores across multiple gold answers
            
        Returns:
            Tuple of (overall_results, example_results)
        """
        assert len(gold_answers) == len(predicted_answers), \
            "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_task_success = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            # Basic task success: check if answer is not empty and matches gold
            if not predicted or predicted.strip() == "":
                task_success = 0.0
            else:
                # Check if predicted answer contains key information from gold answers
                success_scores = []
                for gold in gold_list:
                    # More lenient matching for task success
                    gold_normalized = normalize_answer(gold)
                    pred_normalized = normalize_answer(predicted)
                    
                    # Check for exact match or partial overlap
                    if gold_normalized == pred_normalized:
                        success_scores.append(1.0)
                    elif gold_normalized in pred_normalized or pred_normalized in gold_normalized:
                        success_scores.append(0.8)  # Partial success
                    else:
                        # Check for keyword overlap
                        gold_words = set(gold_normalized.split())
                        pred_words = set(pred_normalized.split())
                        overlap = len(gold_words & pred_words) / len(gold_words) if gold_words else 0
                        success_scores.append(overlap if overlap > 0.5 else 0.0)
                
                task_success = aggregation_fn(success_scores) if success_scores else 0.0
            
            example_eval_results.append({"TaskSuccess": task_success})
            total_task_success += task_success

        avg_task_success = total_task_success / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"TaskSuccess": avg_task_success}

        return pooled_eval_results, example_eval_results

class HippoRAGEvaluator:
    """
    Main evaluator class tương thích với HippoRAG.
    Đánh giá cả retrieval và QA performance với các metric mở rộng.
    """
    
    def __init__(self, include_advanced_metrics: bool = True, test_type: str = "open_end"):
        self.qa_em_evaluator = QAExactMatch()
        self.qa_f1_evaluator = QAF1Score()
        self.retrieval_evaluator = RetrievalRecall()
        self.success_rate_evaluator = SuccessRate()
        self.task_success_evaluator = TaskSuccess()
        
        # Store test type for BLEU metric selection
        self.test_type = test_type
        
        # Advanced metrics
        self.include_advanced_metrics = include_advanced_metrics and ADVANCED_METRICS_AVAILABLE
        if self.include_advanced_metrics:
            # Use BLEU@3 for closed-end questions, BLEU@4 for open/multi questions
            if test_type == "closed_end":
                self.bleu_evaluator = BLEU3Score()
                logger.info("Using BLEU@3 for closed-end questions")
            else:
                self.bleu_evaluator = BLEU4Score()
                logger.info("Using BLEU@4 for open/multi questions")
            
            self.meteor_evaluator = METEORScore()
            self.rouge_l_evaluator = ROUGELScore()
        else:
            if include_advanced_metrics and not ADVANCED_METRICS_AVAILABLE:
                logger.warning("Advanced metrics requested but not available. Install: pip install nltk rouge-score")
        
    def load_test_data(self, file_path: str) -> List[Dict]:
        """Load test data từ JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_gold_docs(self, question_data: Dict) -> List[str]:
        """Extract gold documents từ test data"""
        gold_docs = []
        
        if 'paragraphs' in question_data:
            for paragraph in question_data['paragraphs']:
                if paragraph.get('is_supporting', True):
                    # Chỉ lấy text content để match với passage nodes trong knowledge graph
                    text = paragraph.get('text', '').strip()
                    if text:
                        gold_docs.append(text)
        
        return gold_docs
    
    def evaluate_single_question(self, 
                                rag_system, 
                                question_data: Dict,
                                include_retrieval: bool = True) -> Dict[str, Any]:
        """
        Evaluate một câu hỏi với tất cả metrics.
        
        Args:
            rag_system: Hệ thống RAG
            question_data: Dữ liệu câu hỏi
            include_retrieval: Có đánh giá retrieval không
            
        Returns:
            Dictionary chứa kết quả đánh giá
        """
        question_id = question_data['id']
        question = question_data['question']
        ground_truths = question_data['answer']
        answerable = question_data.get('answerable', True)
        
        # Extract gold docs for retrieval evaluation
        gold_docs = self.extract_gold_docs(question_data) if include_retrieval else None
        
        # Reset timing cho query riêng lẻ
        if hasattr(rag_system, 'reset_timing'):
            rag_system.reset_timing()
        
        # Timing for the entire query process
        start_time = time.time()
        
        # Query hệ thống RAG - SINGLE QUESTION để đo timing chính xác
        try:
            if hasattr(rag_system, 'rag_qa'):
                # Use rag_qa method of HippoRAGComplete - returns 3 values
                # IMPORTANT: Only pass 1 question to get accurate timing
                query_solutions, response_messages, metadata = rag_system.rag_qa([question])
                if query_solutions:
                    predicted_answer = query_solutions[0].answer
                    retrieved_docs = query_solutions[0].docs
                else:
                    predicted_answer = ""
                    retrieved_docs = []
            else:
                # Fallback
                predicted_answer = rag_system.query(question)
                retrieved_docs = []
        except Exception as e:
            logger.error(f"Error querying question {question_id}: {e}")
            predicted_answer = ""
            retrieved_docs = []
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Get per-query timing instead of accumulated timing
        if hasattr(rag_system, 'get_last_query_timing'):
            per_query_timing = rag_system.get_last_query_timing()
        else:
            # Fallback to old method
            per_query_timing = {
                'ppr_time': getattr(rag_system, 'ppr_time', 0),
                'rerank_time': getattr(rag_system, 'rerank_time', 0),
                'retrieval_time': getattr(rag_system, 'all_retrieval_time', 0)
            }
        
        # Calculate QA metrics
        qa_em_result, _ = self.qa_em_evaluator.calculate_metric_scores([ground_truths], [predicted_answer])
        qa_f1_result, _ = self.qa_f1_evaluator.calculate_metric_scores([ground_truths], [predicted_answer])
        
        # Calculate Success Rate and Task Success
        success_rate_result, _ = self.success_rate_evaluator.calculate_metric_scores([ground_truths], [predicted_answer])
        task_success_result, _ = self.task_success_evaluator.calculate_metric_scores([ground_truths], [predicted_answer])
        
        # Calculate advanced QA metrics
        advanced_qa_metrics = {}
        if self.include_advanced_metrics:
            bleu_result, _ = self.bleu_evaluator.calculate_metric_scores([ground_truths], [predicted_answer])
            meteor_result, _ = self.meteor_evaluator.calculate_metric_scores([ground_truths], [predicted_answer])
            rouge_l_result, _ = self.rouge_l_evaluator.calculate_metric_scores([ground_truths], [predicted_answer])
            
            advanced_qa_metrics = {**bleu_result, **meteor_result, **rouge_l_result}
        
        # Calculate retrieval metrics
        retrieval_result = {}
        if include_retrieval and gold_docs and retrieved_docs:
            retrieval_result, _ = self.retrieval_evaluator.calculate_metric_scores(
                [gold_docs], [retrieved_docs], k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            )
        
        return {
            "question_id": question_id,
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truths,
            "answerable": answerable,
            "qa_metrics": {**qa_em_result, **qa_f1_result, **success_rate_result, **task_success_result},
            "advanced_qa_metrics": advanced_qa_metrics,
            "retrieval_metrics": retrieval_result,
            "timing": {
                "total_time": query_time,
                "ppr_time": per_query_timing['ppr_time'],
                "rerank_time": per_query_timing['rerank_time'],
                "retrieval_time": per_query_timing['retrieval_time']
            },
            "retrieved_docs": retrieved_docs[:10],  # Top 10 for analysis
            "gold_docs": gold_docs
        }
    
    def evaluate_dataset(self, 
                        rag_system, 
                        test_file: str,
                        include_retrieval: bool = True,
                        max_questions: Optional[int] = None) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Evaluate toàn bộ dataset.
        
        Args:
            rag_system: Hệ thống RAG
            test_file: File test data
            include_retrieval: Có đánh giá retrieval không
            max_questions: Giới hạn số câu hỏi (None = tất cả)
            
        Returns:
            Tuple of (detailed_results, aggregate_metrics)
        """
        test_data = self.load_test_data(test_file)
        
        if max_questions:
            test_data = test_data[:max_questions]
            
        logger.info(f"Evaluating {len(test_data)} questions from {test_file}")
        
        detailed_results = []
        
        for i, question_data in enumerate(test_data):
            logger.info(f"Processing question {i+1}/{len(test_data)}: {question_data['id']}")
            
            result = self.evaluate_single_question(
                rag_system, question_data, include_retrieval
            )
            detailed_results.append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(detailed_results, include_retrieval)
        
        return detailed_results, aggregate_metrics
    
    def calculate_aggregate_metrics(self, 
                                  results: List[Dict], 
                                  include_retrieval: bool = True) -> Dict[str, float]:
        """Calculate aggregate metrics từ detailed results"""
        if not results:
            return {}
        
        # QA metrics
        qa_metrics = {}
        qa_em_scores = [r["qa_metrics"]["ExactMatch"] for r in results]
        qa_f1_scores = [r["qa_metrics"]["F1"] for r in results]
        success_rate_scores = [r["qa_metrics"].get("SuccessRate_5rounds", 0.0) for r in results]
        task_success_scores = [r["qa_metrics"].get("TaskSuccess", 0.0) for r in results]
        
        qa_metrics["avg_exact_match"] = np.mean(qa_em_scores)
        qa_metrics["avg_f1_score"] = np.mean(qa_f1_scores)
        qa_metrics["avg_success_rate_5rounds"] = np.mean(success_rate_scores)
        qa_metrics["avg_task_success"] = np.mean(task_success_scores)
        
        # Advanced QA metrics
        advanced_qa_metrics = {}
        if self.include_advanced_metrics and results[0].get("advanced_qa_metrics"):
            # BLEU (either BLEU@3 or BLEU@4 depending on test type)
            if self.test_type == "closed_end":
                bleu3_scores = [r["advanced_qa_metrics"].get("BLEU@3", 0.0) for r in results]
                advanced_qa_metrics["avg_bleu3"] = np.mean(bleu3_scores)
            else:
                bleu4_scores = [r["advanced_qa_metrics"].get("BLEU@4", 0.0) for r in results]
                advanced_qa_metrics["avg_bleu4"] = np.mean(bleu4_scores)
            
            # METEOR
            meteor_scores = [r["advanced_qa_metrics"].get("METEOR", 0.0) for r in results]
            advanced_qa_metrics["avg_meteor"] = np.mean(meteor_scores)
            
            # ROUGE-L
            rouge_l_scores = [r["advanced_qa_metrics"].get("ROUGE-L", 0.0) for r in results]
            advanced_qa_metrics["avg_rouge_l"] = np.mean(rouge_l_scores)
        
        # Retrieval metrics
        retrieval_metrics = {}
        if include_retrieval:
            # Get all recall@k keys
            recall_keys = set()
            for r in results:
                if r["retrieval_metrics"]:
                    recall_keys.update(r["retrieval_metrics"].keys())
            
            for key in recall_keys:
                scores = [r["retrieval_metrics"].get(key, 0.0) for r in results 
                         if r["retrieval_metrics"]]
                if scores:
                    retrieval_metrics[f"avg_{key.lower()}"] = np.mean(scores)
        
        # Timing metrics
        timing_metrics = {}
        total_times = [r["timing"]["total_time"] for r in results]
        timing_metrics["avg_total_time"] = np.mean(total_times)
        timing_metrics["total_processing_time"] = np.sum(total_times)
        
        # Combine all metrics
        all_metrics = {**qa_metrics, **retrieval_metrics, **timing_metrics, **advanced_qa_metrics}
        
        # Round to 4 decimal places
        return {k: round(v, 4) for k, v in all_metrics.items()}
    
    def save_results(self, 
                    detailed_results: List[Dict], 
                    aggregate_metrics: Dict[str, float],
                    output_file: str):
        """Save evaluation results to JSON file"""
        output_data = {
            "aggregate_metrics": aggregate_metrics,
            "detailed_results": detailed_results,
            "evaluation_info": {
                "total_questions": len(detailed_results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluator": "HippoRAG Compatible Evaluator"
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, aggregate_metrics: Dict[str, float]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("HIPPORAG EVALUATION SUMMARY")
        print("="*60)
        
        # QA Metrics
        print("\nQA METRICS:")
        print(f"  Exact Match: {aggregate_metrics.get('avg_exact_match', 0):.4f}")
        print(f"  F1 Score:    {aggregate_metrics.get('avg_f1_score', 0):.4f}")
        
        # Advanced QA Metrics
        if any(key.startswith('avg_bleu') or key.startswith('avg_meteor') or key.startswith('avg_rouge_l') 
               for key in aggregate_metrics.keys()):
            print("\nADVANCED QA METRICS:")
            if 'avg_bleu3' in aggregate_metrics:
                print(f"  BLEU@3:      {aggregate_metrics['avg_bleu3']:.4f}")
            if 'avg_bleu4' in aggregate_metrics:
                print(f"  BLEU@4:      {aggregate_metrics['avg_bleu4']:.4f}")
            if 'avg_meteor' in aggregate_metrics:
                print(f"  METEOR:      {aggregate_metrics['avg_meteor']:.4f}")
            if 'avg_rouge_l' in aggregate_metrics:
                print(f"  ROUGE-L:     {aggregate_metrics['avg_rouge_l']:.4f}")
        
        # Retrieval Metrics
        print("\nRETRIEVAL METRICS:")
        for key, value in aggregate_metrics.items():
            if key.startswith('avg_recall@'):
                k = key.replace('avg_recall@', '')
                print(f"  Recall@{k:>3}: {value:.4f}")
        
        # Timing Metrics
        print("\nTIMING METRICS:")
        print(f"  Avg Time per Question: {aggregate_metrics.get('avg_total_time', 0):.2f}s")
        print(f"  Total Processing Time: {aggregate_metrics.get('total_processing_time', 0):.2f}s")
        
        print("="*60)

def main():
    """Demo function"""
    print("HippoRAG Compatible Evaluation System")
    print("Use this module to evaluate your RAG system with HippoRAG metrics")

if __name__ == "__main__":
    main() 