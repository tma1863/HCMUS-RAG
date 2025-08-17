#!/usr/bin/env python3
"""
Test HippoRAG with QA Dataset
Quick test script to evaluate HippoRAG performance on course data
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from docker_config import DockerConfig  # Use docker config for container environment
    from ollama_client import OllamaClient
    from hipporag_complete import HippoRAGComplete
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please ensure all components are available")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_qa_data(dataset_name="AM", qa_dir="QA for testing"):
    """Load corpus and test data for a specific dataset"""
    
    dataset_dir = Path(qa_dir) / dataset_name
    
    # Load corpus
    corpus_file = dataset_dir / f"{dataset_name}_corpus.json"
    corpus_data = []
    if corpus_file.exists():
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        logger.info(f"Loaded {len(corpus_data)} corpus documents")
    
    # Load test questions  
    test_files = {
        'closed_end': dataset_dir / f"{dataset_name}_closed_end.json",
        'opened_end': dataset_dir / f"{dataset_name}_opened_end.json", 
        'multihop': dataset_dir / f"{dataset_name}_multihop.json"
    }
    
    test_data = {}
    for test_type, test_file in test_files.items():
        if test_file.exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            test_data[test_type] = questions
            logger.info(f"Loaded {len(questions)} {test_type} questions")
    
    return corpus_data, test_data

def extract_corpus_documents(corpus_data):
    """Extract text documents from corpus data"""
    documents = []
    
    for item in corpus_data:
        if isinstance(item, dict):
            # Try different text fields
            text = item.get('text', '') or item.get('paragraph_text', '') or item.get('content', '')
            title = item.get('title', '')
            
            # Combine title and text
            if title and text:
                doc = f"{title}\n\n{text}"
            elif text:
                doc = text
            elif title:
                doc = title
            else:
                # Fallback: combine all string values
                doc = ' '.join([str(v) for v in item.values() if isinstance(v, str) and len(v) > 10])
            
            if doc.strip():
                documents.append(doc.strip())
    
    return documents

def run_qa_test(dataset_name="AM", test_type="closed_end", max_questions=5):
    """Run QA test on specified dataset"""
    
    print(f"\nTesting HippoRAG on {dataset_name} - {test_type}")
    print("="*60)
    
    # Load data
    corpus_data, test_data = load_qa_data(dataset_name)
    
    if not corpus_data:
        print(f"No corpus data found for {dataset_name}")
        return
    
    if test_type not in test_data:
        print(f"No {test_type} test data found for {dataset_name}")
        return
    
    # Extract documents
    documents = extract_corpus_documents(corpus_data)
    questions_data = test_data[test_type][:max_questions]
    
    print(f"Corpus: {len(documents)} documents")
    print(f"Questions: {len(questions_data)} {test_type} questions")
    
    # Initialize HippoRAG with docker configuration
    save_dir = f"outputs/qa_test_{dataset_name}_{test_type}"
    docker_config = DockerConfig()
    hipporag = HippoRAGComplete(global_config=docker_config, save_dir=save_dir)
    
    # Index documents
    print(f"\nIndexing documents...")
    start_time = time.time()
    hipporag.index(documents)
    index_time = time.time() - start_time
    print(f"Indexing completed in {index_time:.1f}s")
    
    # Test questions
    print(f"\nTesting questions...")
    results = []
    
    for i, qa_item in enumerate(questions_data, 1):
        question = qa_item.get('question', '')
        expected_answer = qa_item.get('answer', [''])
        
        if isinstance(expected_answer, str):
            expected_answer = [expected_answer]
        
        print(f"\n[{i}/{len(questions_data)}] {question}")
        
        try:
            # Get answer from HippoRAG
            start_time = time.time()
            query_solutions, _, _ = hipporag.rag_qa([question])
            query_time = time.time() - start_time
            
            if query_solutions:
                answer = query_solutions[0].answer
                retrieved_docs = query_solutions[0].docs[:3]  # Top 3 docs
                doc_scores = query_solutions[0].doc_scores[:3]
                
                print(f"Answer: {answer}")
                print(f"Expected: {' | '.join(expected_answer)}")
                print(f"Time: {query_time:.2f}s")
                
                # Simple evaluation
                answer_lower = answer.lower()
                match_found = any(exp.lower() in answer_lower for exp in expected_answer)
                
                result = {
                    'question': question,
                    'expected_answer': expected_answer,
                    'generated_answer': answer,
                    'match_found': match_found,
                    'query_time': query_time,
                    'top_doc_score': doc_scores[0] if doc_scores else 0
                }
                results.append(result)
                
                print(f"Match: {'YES' if match_found else 'NO'}")
                
                # Show top retrieved document
                if retrieved_docs:
                    print(f"Top doc: {retrieved_docs[0][:100]}...")
                
            else:
                print("No answer generated")
                
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"Question processing error: {e}")
    
    # Summary
    print(f"\nRESULTS SUMMARY:")
    print("="*50)
    
    if results:
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r['match_found'])
        avg_time = sum(r['query_time'] for r in results) / total_questions
        avg_score = sum(r['top_doc_score'] for r in results) / total_questions
        
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Accuracy: {correct_answers/total_questions:.2%}")
        print(f"Avg Query Time: {avg_time:.2f}s")
        print(f"Avg Doc Score: {avg_score:.4f}")
        
        # Save results
        results_file = f"{save_dir}/qa_test_results.json"
        os.makedirs(save_dir, exist_ok=True)
        
        summary = {
            'dataset': dataset_name,
            'test_type': test_type,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': correct_answers/total_questions,
            'avg_query_time': avg_time,
            'avg_doc_score': avg_score,
            'detailed_results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {results_file}")
    
    else:
        print("No results to summarize")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test HippoRAG with QA Dataset')
    parser.add_argument('--dataset', default='AM', choices=['AM', 'DS', 'MCS'],
                       help='Dataset to test (default: AM)')
    parser.add_argument('--test_type', default='closed_end', 
                       choices=['closed_end', 'opened_end', 'multihop'],
                       help='Test type (default: closed_end)')
    parser.add_argument('--max_questions', type=int, default=5,
                       help='Max questions to test (default: 5)')
    
    args = parser.parse_args()
    
    print("HippoRAG QA Testing")
    print(f"Dataset: {args.dataset}")
    print(f"Test Type: {args.test_type}")
    print(f"Max Questions: {args.max_questions}")
    
    try:
        run_qa_test(args.dataset, args.test_type, args.max_questions)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        logger.error(f"Test error: {e}")
        raise

if __name__ == "__main__":
    main() 