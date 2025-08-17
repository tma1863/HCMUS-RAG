#!/usr/bin/env python3
"""
HippoRAG Gradio Dashboard  
========================
Gradio-based web interface for HippoRAG - much simpler than FastAPI!

Features:
- Beautiful Gradio UI with chatbot interface
- Real-time graph statistics
- Debug reranking visualization  
- Session management & export
- One-click dataset setup
- Mobile-friendly responsive design

Usage:
    python hipporag_gradio_app.py
    
Access: http://localhost:7860
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import gradio as gr
import pandas as pd

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
hipporag_instances = {}  # Cache for different datasets
current_session = {"dataset": None, "session_id": None, "hipporag": None, "debug": False}
session_history = []

def initialize_app():
    """Initialize configuration and setup"""
    global config
    
    try:
        config = DockerConfig()
        logger.info("Configuration loaded successfully")
        
        # Create required directories
        os.makedirs("outputs/gradio_sessions", exist_ok=True)
        
        return "Application initialized successfully!"
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return f"Initialization failed: {e}"

def get_or_create_hipporag(dataset: str, debug_reranking: bool = False) -> Tuple[HippoRAGComplete, str]:
    """Get or create HippoRAG instance for dataset"""
    cache_key = f"{dataset}_{debug_reranking}"
    
    try:
        if cache_key not in hipporag_instances:
            logger.info(f"Creating new HippoRAG instance for {dataset}")
            
            save_dir = f"outputs/gradio_sessions/{dataset}_corpus_shared"
            hipporag = HippoRAGComplete(global_config=config, save_dir=save_dir)
            
            if debug_reranking:
                patch_rerank_with_detailed_logging(hipporag)
            
            hipporag_instances[cache_key] = hipporag
            
        return hipporag_instances[cache_key], "HippoRAG instance ready"
        
    except Exception as e:
        logger.error(f"Error creating HippoRAG instance: {e}")
        return None, f"Error: {e}"

def start_session(dataset: str, debug_reranking: bool, force_rebuild: bool) -> Tuple[str, str, str]:
    """Start new session with dataset"""
    global current_session, session_history
    
    try:
        if not dataset:
            return "Please select a dataset", "", ""
        
        # Clear previous session
        session_history = []
        
        # Create session ID
        session_id = f"{dataset}_{int(time.time())}"
        
        # Get HippoRAG instance
        hipporag, status = get_or_create_hipporag(dataset, debug_reranking)
        if hipporag is None:
            return status, "", ""
        
        # Load and index corpus
        status_msg = f"Starting session for {dataset}...\n"
        
        docs = load_corpus(dataset, "QA for testing")
        if not docs:
            return f"No documents found for dataset {dataset}", "", ""
        
        status_msg += f"Loaded {len(docs)} documents\n"
        
        # Check if corpus needs indexing
        corpus_indexed_file = os.path.join(hipporag.save_dir, f"{dataset}_corpus_indexed.flag")
        
        if not os.path.exists(corpus_indexed_file) or force_rebuild:
            status_msg += "Indexing corpus (this may take a few minutes)...\n"
            
            start_time = time.time()
            hipporag.index(docs)
            index_time = time.time() - start_time
            
            status_msg += f"Corpus indexed in {index_time:.1f}s\n"
            
            # Create flag file
            os.makedirs(hipporag.save_dir, exist_ok=True)
            with open(corpus_indexed_file, 'w') as f:
                f.write(f"Corpus {dataset} indexed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Documents: {len(docs)}\n")
                f.write(f"Index time: {index_time:.1f}s\n")
        else:
            status_msg += "Using existing corpus index\n"
        
        # Update current session
        current_session = {
            "dataset": dataset,
            "session_id": session_id,
            "hipporag": hipporag,
            "debug": debug_reranking,
            "created_at": datetime.now().isoformat()
        }
        
        # Get graph statistics
        graph_stats = get_graph_statistics()
        
        status_msg += f"Session {session_id} started successfully!\n"
        status_msg += "You can now ask questions in the chatbot below."
        
        return status_msg, f"Active Session: {session_id} ({dataset})", graph_stats
        
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return f"Error starting session: {e}", "", ""

def get_graph_statistics() -> str:
    """Get and format graph statistics"""
    if not current_session["hipporag"]:
        return "No active session"
    
    try:
        hipporag = current_session["hipporag"]
        
        # Get graph stats
        stats = {}
        if hasattr(hipporag, 'get_graph_info'):
            stats = hipporag.get_graph_info()
        else:
            # Fallback stats
            stats = {
                "nodes": hipporag.graph.vcount() if hasattr(hipporag, 'graph') else 0,
                "edges": hipporag.graph.ecount() if hasattr(hipporag, 'graph') else 0,
                "chunks": hipporag.chunk_embedding_store.get_passage_count() if hasattr(hipporag, 'chunk_embedding_store') else 0,
                "entities": hipporag.entity_embedding_store.get_entity_count() if hasattr(hipporag, 'entity_embedding_store') else 0,
                "facts": hipporag.fact_embedding_store.get_fact_count() if hasattr(hipporag, 'fact_embedding_store') else 0
            }
        
        # Format stats
        stats_text = "**Knowledge Graph Statistics**\n\n"
        stats_text += f"**Nodes:** {stats.get('nodes', 0):,}\n"
        stats_text += f"**Edges:** {stats.get('edges', 0):,}\n"
        stats_text += f"**Chunks:** {stats.get('chunks', 0):,}\n"
        stats_text += f"**Entities:** {stats.get('entities', 0):,}\n"
        stats_text += f"**Facts:** {stats.get('facts', 0):,}\n"
        stats_text += f"\n**Last Updated:** {datetime.now().strftime('%H:%M:%S')}"
        
        return stats_text
        
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        return f"Error getting statistics: {e}"

def ask_question(message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """Process question and return answer"""
    global session_history
    
    if not current_session["hipporag"]:
        return "Please start a session first!", history
    
    if not message.strip():
        return "", history
    
    try:
        hipporag = current_session["hipporag"]
        
        # Ensure graph is initialized before processing
        if hipporag.graph is None:
            logger.info("Graph not loaded, initializing...")
            hipporag.graph = hipporag.initialize_graph()
            logger.info("Graph initialized successfully")
        
        # Process question
        start_time = time.time()
        query_solutions, all_responses, all_metadata = hipporag.rag_qa([message])
        processing_time = time.time() - start_time
        
        query_solution = query_solutions[0]
        answer = query_solution.answer
        retrieved_docs = query_solution.docs[:3]  # Top 3 docs
        doc_scores = query_solution.doc_scores[:3] if hasattr(query_solution, 'doc_scores') else []
        
        # Format response with metadata
        response = f"**Answer:** {answer}\n\n"
        
        if retrieved_docs:
            response += f"**Retrieved Context ({len(retrieved_docs)} docs):**\n"
            for i, (doc, score) in enumerate(zip(retrieved_docs, doc_scores + [0.0] * len(retrieved_docs))):
                response += f"\n**[{i+1}]** (Score: {score:.4f})\n"
                response += f"{doc[:200]}{'...' if len(doc) > 200 else ''}\n"
        
        response += f"\n*Processing time: {processing_time:.2f}s*"
        
        # Add to session history
        session_history.append({
            "question": message,
            "answer": answer,
            "processing_time": processing_time,
            "retrieved_docs": len(retrieved_docs),
            "timestamp": datetime.now().isoformat()
        })
        
        # Update chat history
        history.append([message, response])
        
        return "", history
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        error_response = f"Error processing question: {e}"
        history.append([message, error_response])
        return "", history

def get_session_summary() -> str:
    """Get session summary"""
    if not current_session["dataset"]:
        return "No active session"
    
    try:
        summary = f"**Session Summary**\n\n"
        summary += f"**Dataset:** {current_session['dataset']}\n"
        summary += f"**Session ID:** {current_session['session_id']}\n"
        summary += f"**Debug Mode:** {'Enabled' if current_session['debug'] else 'Disabled'}\n"
        summary += f"**Created:** {current_session['created_at']}\n"
        summary += f"**Questions Asked:** {len(session_history)}\n"
        
        if session_history:
            avg_time = sum(q["processing_time"] for q in session_history) / len(session_history)
            summary += f"**Avg Response Time:** {avg_time:.2f}s\n"
            summary += f"**Total Docs Retrieved:** {sum(q['retrieved_docs'] for q in session_history)}\n"
        
        return summary
        
    except Exception as e:
        return f"Error getting summary: {e}"

def export_session() -> str:
    """Export session data to JSON"""
    if not current_session["dataset"]:
        return "No active session to export"
    
    try:
        export_data = {
            "session_info": current_session.copy(),
            "questions_history": session_history,
            "export_timestamp": datetime.now().isoformat(),
            "total_questions": len(session_history)
        }
        
        # Remove hipporag instance from export (not serializable)
        export_data["session_info"].pop("hipporag", None)
        
        filename = f"outputs/gradio_sessions/session_{current_session['session_id']}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return f"Session exported to: {filename}"
        
    except Exception as e:
        logger.error(f"Error exporting session: {e}")
        return f"Export failed: {e}"

def clear_session() -> Tuple[str, str, List, str]:
    """Clear current session"""
    global current_session, session_history
    
    current_session = {"dataset": None, "session_id": None, "hipporag": None, "debug": False}
    session_history = []
    
    return "Session cleared", "No active session", [], "No session data"

def create_gradio_interface():
    """Create Gradio interface"""
    
    # Initialize app
    init_status = initialize_app()
    logger.info(f"App initialization: {init_status}")
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .chatbot {
        height: 600px !important;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    """
    
    with gr.Blocks(
        title="HippoRAG Dashboard",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        
        gr.Markdown("""
        # HippoRAG Interactive Dashboard
        
        **Knowledge Graph-based RAG System** - Ask questions about your datasets!
        
        Choose a dataset, start a session, and begin asking questions. The system will use knowledge graphs to provide accurate answers with retrieved context.
        """)
        
        with gr.Tabs():
            
            # === SESSION SETUP TAB ===
            with gr.TabItem("Session Setup"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Dataset Selection")
                        dataset_dropdown = gr.Dropdown(
                            choices=["AM", "DS", "MCS"],
                            label="Select Dataset",
                            info="AM = Applied Math, DS = Data Science, MCS = Master Computer Science"
                        )
                        
                        debug_checkbox = gr.Checkbox(
                            label="Enable Debug Reranking",
                            info="Show detailed reranking process (slower but more informative)"
                        )
                        
                        force_rebuild_checkbox = gr.Checkbox(
                            label="Force Corpus Rebuild",
                            info="Rebuild corpus index from scratch (takes longer)"
                        )
                        
                        start_btn = gr.Button("Start Session", variant="primary", size="lg")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### Session Status")
                        status_text = gr.Textbox(
                            label="Status",
                            placeholder="Select dataset and click 'Start Session'",
                            lines=8,
                            interactive=False
                        )
                        
                        session_info = gr.Textbox(
                            label="Current Session",
                            placeholder="No active session",
                            interactive=False
                        )
                
                # Connect start session
                start_btn.click(
                    start_session,
                    inputs=[dataset_dropdown, debug_checkbox, force_rebuild_checkbox],
                    outputs=[status_text, session_info, gr.Textbox(visible=False)]  # Hidden graph stats
                )
            
            # === INTERACTIVE Q&A TAB ===
            with gr.TabItem("Interactive Q&A"):
                gr.Markdown("### Ask Questions")
                
                chatbot = gr.Chatbot(
                    label="HippoRAG Assistant",
                    height=600,
                    show_label=True,
                    avatar_images=None
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about the selected dataset...",
                        lines=2,
                        scale=4
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)
                
                gr.Markdown("**Instructions:**")
                gr.Markdown("1. Start a session in the 'Session Setup' tab first")
                gr.Markdown("2. Type your question in the text box above")
                gr.Markdown("3. Click 'Ask' or press Enter to get an answer")
                gr.Markdown("4. View retrieved context and processing time in the response")
                
                # Connect question asking
                question_input.submit(
                    ask_question,
                    inputs=[question_input, chatbot],
                    outputs=[question_input, chatbot]
                )
                
                ask_btn.click(
                    ask_question,
                    inputs=[question_input, chatbot],
                    outputs=[question_input, chatbot]
                )
            
            # === STATISTICS TAB ===
            with gr.TabItem("Statistics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Knowledge Graph Statistics")
                        graph_stats = gr.Markdown("Start a session to view graph statistics")
                        
                        refresh_stats_btn = gr.Button("Refresh Statistics", variant="secondary")
                        
                        refresh_stats_btn.click(
                            get_graph_statistics,
                            outputs=graph_stats
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Session Summary")
                        session_summary = gr.Markdown("No active session")
                        
                        refresh_summary_btn = gr.Button("Refresh Summary", variant="secondary")
                        
                        refresh_summary_btn.click(
                            get_session_summary,
                            outputs=session_summary
                        )
            
            # === SESSION MANAGEMENT TAB ===
            with gr.TabItem("Session Management"):
                gr.Markdown("### Session Actions")
                
                with gr.Row():
                    export_btn = gr.Button("Export Session", variant="secondary")
                    clear_btn = gr.Button("Clear Session", variant="secondary")
                
                action_status = gr.Textbox(
                    label="Action Status",
                    placeholder="Action results will appear here",
                    lines=3,
                    interactive=False
                )
                
                gr.Markdown("### Session History")
                history_display = gr.JSON(label="Question History", visible=False)
                
                # Connect session management
                export_btn.click(
                    export_session,
                    outputs=action_status
                )
                
                clear_btn.click(
                    clear_session,
                    outputs=[action_status, session_info, chatbot, session_summary]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("**HippoRAG Dashboard** - Powered by Gradio")
    
    return demo

if __name__ == "__main__":
    # Create and launch Gradio app
    demo = create_gradio_interface()
    
    # Launch configuration
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Keep False for Docker
        debug=False,  # Disable debug to avoid JSON schema errors
        show_error=True,
        quiet=True,  # Reduce noise
        prevent_thread_lock=False,
        show_api=False  # Disable API docs to avoid schema errors
    ) 