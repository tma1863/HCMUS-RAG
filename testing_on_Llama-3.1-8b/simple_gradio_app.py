#!/usr/bin/env python3
"""
Simple HippoRAG Gradio Dashboard for Testing
==========================================
Simplified version to avoid JSON schema issues
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import gradio as gr

# Import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.docker_config import DockerConfig
    from hipporag_complete import HippoRAG as HippoRAGComplete
    from main_hipporag import load_corpus, patch_rerank_with_detailed_logging
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
config = None
hipporag_instances = {}
current_session = {"dataset": None, "hipporag": None, "debug_mode": False, "last_debug_info": None}

def patch_rerank_for_dashboard(hipporag_instance):
    """
    🔥 Enhanced rerank patch for dashboard - captures AND returns debug info
    """
    import types
    import numpy as np
    
    def dashboard_debug_rerank_facts(self, query: str, query_fact_scores: np.ndarray):
        """Dashboard version của rerank_facts - capture debug info for UI"""
        
        debug_info = {
            'query': query,
            'query_fact_scores_stats': {
                'shape': str(query_fact_scores.shape),
                'min': float(query_fact_scores.min()) if len(query_fact_scores) > 0 else 0,
                'max': float(query_fact_scores.max()) if len(query_fact_scores) > 0 else 0,
                'mean': float(query_fact_scores.mean()) if len(query_fact_scores) > 0 else 0
            }
        }
        
        # load args
        link_top_k: int = self.global_config.linking_top_k
        debug_info['linking_top_k'] = link_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            debug_info['error'] = "No facts available for reranking"
            current_session["last_debug_info"] = debug_info
            return [], [], {'dashboard_debug': debug_info}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
                debug_info['selection_method'] = f"Using all {len(query_fact_scores)} facts (fewer than top_k={link_top_k})"
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                debug_info['selection_method'] = f"Selected top {link_top_k} facts from {len(query_fact_scores)} total"
                
            debug_info['candidate_indices'] = candidate_fact_indices[:10]  # Limit for display
            debug_info['candidate_scores'] = [float(query_fact_scores[i]) for i in candidate_fact_indices[:10]]
            
            # Get the actual fact IDs and content
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # Store Query to Triple (Top-5)
            debug_info['query_to_triple'] = []
            for i, fact in enumerate(candidate_facts[:5]):
                score = float(query_fact_scores[candidate_fact_indices[i]])
                debug_info['query_to_triple'].append({
                    'rank': i+1,
                    'score': score,
                    'fact': str(fact)[:200] + "..." if len(str(fact)) > 200 else str(fact)
                })
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            # Store Filtered Triple
            debug_info['filtered_triple'] = []
            for i, fact in enumerate(top_k_facts):
                debug_info['filtered_triple'].append({
                    'rank': i+1,
                    'fact': str(fact)[:200] + "..." if len(str(fact)) > 200 else str(fact)
                })
            
            # Summary stats
            debug_info['reranking_summary'] = {
                'facts_before': len(candidate_facts),
                'facts_after': len(top_k_facts),
                'filtered_out': len(candidate_facts) - len(top_k_facts),
                'retention_rate': len(top_k_facts)/len(candidate_facts)*100 if len(candidate_facts) > 0 else 0
            }
            
            debug_info['reranker_metadata'] = str(reranker_dict)[:500] + "..." if len(str(reranker_dict)) > 500 else str(reranker_dict)
            
            if len(top_k_facts) == 0:
                debug_info['warning'] = "CRITICAL: Reranking returned ZERO facts! This will cause fallback to DPR."
            
            # Store in session for dashboard access
            current_session["last_debug_info"] = debug_info
            
            return top_k_fact_indices, top_k_facts, {'dashboard_debug': debug_info}
            
        except Exception as e:
            debug_info['error'] = str(e)
            current_session["last_debug_info"] = debug_info
            return [], [], {'dashboard_debug': debug_info}

    # Patch the method
    hipporag_instance.rerank_facts = types.MethodType(dashboard_debug_rerank_facts, hipporag_instance)
    logger.info("🔧 Dashboard debug rerank patch applied successfully!")

def initialize_app():
    """Initialize configuration and setup"""
    global config
    
    try:
        config = DockerConfig()
        logger.info("Configuration loaded successfully")
        os.makedirs("outputs/gradio_sessions", exist_ok=True)
        return "✅ Application initialized successfully!"
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return f"❌ Initialization failed: {e}"

def setup_dataset(dataset: str, debug_mode: bool) -> str:
    """Setup dataset and return status"""
    global current_session
    
    try:
        if not dataset:
            return "❌ Please select a dataset"
        
        # Get HippoRAG instance
        save_dir = f"outputs/gradio_sessions/{dataset}_corpus_shared"
        hipporag = HippoRAGComplete(global_config=config, save_dir=save_dir)
        
        if debug_mode:
            patch_rerank_for_dashboard(hipporag)  # 🔥 Use dashboard-specific debug patch
            current_session["debug_mode"] = True
        else:
            current_session["debug_mode"] = False
        
        # Load and index corpus
        docs = load_corpus(dataset, "QA for testing")
        if not docs:
            return f"❌ No documents found for dataset {dataset}"
        
        # 🔥 SMART CORPUS INDEXING: Same logic as main_hipporag.py
        corpus_indexed_file = os.path.join(save_dir, f"{dataset}_corpus_indexed.flag")
        
        # Initialize variables  
        index_time = 0
        needs_indexing = True
        status_msg = ""
        
        # Check if corpus needs indexing (same logic as main_hipporag.py)
        if os.path.exists(corpus_indexed_file):
            try:
                with open(corpus_indexed_file, 'r') as f:
                    flag_content = f.read().strip()
                    # Extract previously indexed document count
                    for line in flag_content.split('\n'):
                        if line.startswith('Documents:'):
                            prev_doc_count = int(line.split(':')[1].strip())
                            current_doc_count = len(docs)
                            
                            if prev_doc_count == current_doc_count:
                                status = f"✅ Dataset {dataset} already indexed!\n"
                                status += f"📄 {flag_content.replace(chr(10), ' | ')}\n"
                                status += f"📚 {len(docs)} documents available\n"
                                needs_indexing = False
                            else:
                                needs_indexing = True
                                status_msg = f"🔄 Corpus size changed: {prev_doc_count} → {current_doc_count} documents"
                            break
                    else:
                        # Flag file doesn't have document count, need to re-index
                        needs_indexing = True
                        status_msg = "⚠️ Flag file missing document count, performing full indexing"
            except Exception as e:
                needs_indexing = True
                status_msg = f"⚠️ Error reading flag file: {e}, performing full indexing"
        else:
            needs_indexing = True
            status_msg = "📝 No existing corpus found, performing initial indexing"
        
        if needs_indexing:
            start_time = time.time()
            hipporag.index(docs)  # HippoRAG handles incremental automatically
            index_time = time.time() - start_time
            
            # Update flag file with current document count
            os.makedirs(save_dir, exist_ok=True)
            with open(corpus_indexed_file, 'w') as f:
                f.write(f"Corpus {dataset} indexed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Documents: {len(docs)}\n")
                f.write(f"Index time: {index_time:.1f}s\n")
                f.write(f"Incremental indexing: Yes\n")
            
            status = f"✅ Dataset {dataset} setup complete!\n"
            status += f"{status_msg}\n"
            status += f"📚 Indexed {len(docs)} documents in {index_time:.1f}s\n"
        
        # Update session
        current_session["dataset"] = dataset
        current_session["hipporag"] = hipporag
        
        # 🔥 FIX: Get graph stats directly from graph object (like in logs)
        try:
            if hasattr(hipporag, 'graph') and hipporag.graph is not None:
                nodes = hipporag.graph.vcount()
                edges = hipporag.graph.ecount()
                status += f"📊 Graph: {nodes:,} nodes, {edges:,} edges\n"
            else:
                status += f"⚠️ Graph not loaded yet\n"
        except Exception as e:
            logger.warning(f"Could not get graph stats: {e}")
            status += f"⚠️ Graph stats unavailable\n"
            
        status += "💬 Ready to answer questions!"
        return status
        
    except Exception as e:
        logger.error(f"Error setting up dataset: {e}")
        return f"❌ Error: {e}"

def ask_question(question: str) -> Tuple[str, str, str]:
    """
    Process question and return answer - Interactive & Debug Mode
    🔥 Enhanced like CLI --interactive --debug_reranking
    """
    
    if not current_session["hipporag"]:
        return "❌ Please setup a dataset first!", "", ""
    
    if not question.strip():
        return "❌ Please enter a question", "", ""
    
    try:
        hipporag = current_session["hipporag"]
        
        # 🔍 INTERACTIVE MODE: Print question processing like CLI
        logger.info(f"🔍 Processing question: '{question}'")
        
        # Process question
        start_time = time.time()
        query_solutions, all_responses, all_metadata = hipporag.rag_qa([question])
        processing_time = time.time() - start_time
        
        query_solution = query_solutions[0]
        answer = query_solution.answer
        retrieved_docs = query_solution.docs[:5]  # 🔥 TOP 5 DOCS như request
        doc_scores = query_solution.doc_scores[:5] if hasattr(query_solution, 'doc_scores') else [0.0] * len(retrieved_docs)
        
        # 🔥 ENHANCED ANSWER FORMAT (CLI-like)
        response = f"💡 **Answer:** {answer}\n\n"
        response += f"⏱️ *Processing time: {processing_time:.2f}s*\n"
        response += f"📊 *Retrieved {len(retrieved_docs)} documents*\n"
        
        # 🔥 ENHANCED CONTEXT FORMAT (CLI-like with scores)
        context = ""
        if retrieved_docs:
            context = f"📄 **Retrieved Context (Top {len(retrieved_docs)} documents):**\n\n"
            for i, (doc, score) in enumerate(zip(retrieved_docs, doc_scores), 1):
                context += f"**[{i}] Score: {score:.4f}**\n"
                context += f"{doc[:400]}{'...' if len(doc) > 400 else ''}\n\n"
                context += "─" * 60 + "\n\n"
        
        # 🔥 ENHANCED DEBUG RERANKING INFO (from dashboard debug patch)
        debug_info = ""
        
        # Always show basic graph stats
        if hasattr(hipporag, 'graph'):
            debug_info += f"📊 **Graph Stats:**\n"
            debug_info += f"  • Nodes: {hipporag.graph.vcount():,}\n"
            debug_info += f"  • Edges: {hipporag.graph.ecount():,}\n\n"
        
        # Show processing pipeline
        debug_info += f"📈 **Retrieval Stats:**\n"
        debug_info += f"  • Top doc score: {doc_scores[0]:.4f}\n" if doc_scores else ""
        debug_info += f"  • Avg doc score: {sum(doc_scores)/len(doc_scores):.4f}\n" if doc_scores else ""
        debug_info += f"  • Processing pipeline: DPR → Rerank → QA\n\n"
        
        # 🔥 ENHANCED DEBUG INFO from dashboard patch
        if current_session.get("debug_mode") and current_session.get("last_debug_info"):
            last_debug = current_session["last_debug_info"]
            debug_info += f"🔧 **Debug Reranking Info (--debug_reranking mode):**\n\n"
            
            # Query info
            debug_info += f"🔍 **Query:** '{last_debug.get('query', 'N/A')}'\n\n"
            
            # Fact scores stats
            if 'query_fact_scores_stats' in last_debug:
                stats = last_debug['query_fact_scores_stats']
                debug_info += f"📊 **Query Fact Scores:**\n"
                debug_info += f"  • Shape: {stats.get('shape', 'N/A')}\n"
                debug_info += f"  • Min: {stats.get('min', 0):.4f}\n"
                debug_info += f"  • Max: {stats.get('max', 0):.4f}\n"
                debug_info += f"  • Mean: {stats.get('mean', 0):.4f}\n\n"
            
            # Selection method
            if 'selection_method' in last_debug:
                debug_info += f"📝 **Selection:** {last_debug['selection_method']}\n\n"
            
            # Query to Triple (Top-5)
            if 'query_to_triple' in last_debug:
                debug_info += f"🔗 **Query to Triple (Top-5):**\n"
                debug_info += "=" * 50 + "\n"
                for item in last_debug['query_to_triple']:
                    debug_info += f"  {item['rank']}. [Score: {item['score']:.4f}] {item['fact']}\n"
                debug_info += "=" * 50 + "\n\n"
            
            # Filtered Triple
            if 'filtered_triple' in last_debug:
                debug_info += f"🎯 **Filtered Triple ({len(last_debug['filtered_triple'])} facts):**\n"
                debug_info += "=" * 50 + "\n"
                if len(last_debug['filtered_triple']) == 0:
                    debug_info += "  ⚠️ NO FACTS RETURNED AFTER RERANKING!\n"
                else:
                    for item in last_debug['filtered_triple']:
                        debug_info += f"  {item['rank']}. {item['fact']}\n"
                debug_info += "=" * 50 + "\n\n"
            
            # Reranking summary
            if 'reranking_summary' in last_debug:
                summary = last_debug['reranking_summary']
                debug_info += f"📈 **RERANKING SUMMARY:**\n"
                debug_info += f"  • Query to Triple: {summary.get('facts_before', 0)} facts\n"
                debug_info += f"  • Filtered Triple: {summary.get('facts_after', 0)} facts\n"
                debug_info += f"  • Filtered out: {summary.get('filtered_out', 0)} facts\n"
                debug_info += f"  • Retention rate: {summary.get('retention_rate', 0):.1f}%\n\n"
            
            # Warnings
            if 'warning' in last_debug:
                debug_info += f"🚨 **WARNING:** {last_debug['warning']}\n\n"
            
            # Errors
            if 'error' in last_debug:
                debug_info += f"❌ **ERROR:** {last_debug['error']}\n\n"
        else:
            debug_info += f"💡 **Enable Debug Mode** to see detailed reranking information\n"
            debug_info += f"   (Equivalent to CLI --debug_reranking flag)"
        
        return response, context, debug_info
        
    except Exception as e:
        error_msg = f"❌ Error processing question: {e}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()  # 🔥 FULL TRACEBACK like --debug_reranking
        return error_msg, "", f"🔧 **Error Details:**\n{str(e)}"

def create_simple_interface():
    """Create simple Gradio interface"""
    
    # Initialize app
    init_status = initialize_app()
    logger.info(f"App initialization: {init_status}")
    
    with gr.Blocks(title="🦛 HippoRAG Simple Dashboard") as demo:
        
        gr.Markdown("# 🦛 HippoRAG Interactive Dashboard")
        gr.Markdown("Knowledge Graph-based RAG System - **Interactive & Debug Mode** 🔧")
        gr.Markdown("*Equivalent to: `docker exec -it hipporag_app python main_hipporag.py --dataset [DS] --interactive --debug_reranking`*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")
                
                dataset_dropdown = gr.Dropdown(
                    choices=["AM", "DS", "MCS"],
                    label="Select Dataset",
                    value=None
                )
                
                debug_checkbox = gr.Checkbox(
                    label="Enable Debug Mode",
                    value=False
                )
                
                setup_btn = gr.Button("🚀 Setup Dataset", variant="primary")
                
                setup_status = gr.Textbox(
                    label="Setup Status",
                    lines=6,
                    interactive=False,
                    placeholder="Select dataset and click Setup"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### Ask Questions")
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about the dataset...",
                    lines=2
                )
                
                ask_btn = gr.Button("💬 Ask Question", variant="secondary")
                
                # 🔥 ENHANCED OUTPUT - CLI-like Interface
                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="💡 Answer",
                            lines=6,
                            interactive=False
                        )
                        
                        context_output = gr.Textbox(
                            label="📄 Retrieved Context (Top 5 Documents)",
                            lines=8,
                            interactive=False
                        )
                        
                    with gr.Column():
                        debug_output = gr.Textbox(
                            label="🔧 Debug Reranking Info (--debug_reranking)",
                            lines=14,
                            interactive=False,
                            placeholder="Enable debug mode to see reranking details..."
                        )
        
        # Connect events
        setup_btn.click(
            setup_dataset,
            inputs=[dataset_dropdown, debug_checkbox],
            outputs=setup_status
        )
        
        # 🔥 ENHANCED EVENT HANDLERS - 3 outputs for debug mode
        ask_btn.click(
            ask_question,
            inputs=question_input,
            outputs=[answer_output, context_output, debug_output]
        )
        
        question_input.submit(
            ask_question,
            inputs=question_input,
            outputs=[answer_output, context_output, debug_output]
        )
        
        gr.Markdown("---")
        gr.Markdown("**🦛 HippoRAG Interactive Dashboard** - *CLI --interactive --debug_reranking Mode*")
        gr.Markdown("📊 **Features:** Smart Indexing • Top-5 Docs • Debug Reranking • Graph Stats • Real-time Processing")
    
    return demo

if __name__ == "__main__":
    # Create and launch simple app
    demo = create_simple_interface()
    
    # Simple launch configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        quiet=True
    ) 