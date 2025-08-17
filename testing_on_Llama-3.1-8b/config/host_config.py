"""
Host Configuration for HippoRAG
Configuration for running HippoRAG from host machine (outside Docker)
Uses localhost URLs instead of Docker service names
"""

import os
from dataclasses import dataclass
from config import Config

@dataclass
class HostConfig(Config):
    """Configuration for running HippoRAG from host machine"""
    
    # LLM Configuration - use localhost instead of Docker service name
    llm_name: str = "llama3:8b"
    llm_base_url: str = "http://localhost:11434/api/generate"  # Use native Ollama endpoint
    
    # OpenIE Configuration - use native Ollama endpoint for SimpleOpenIE
    openie_llm_name: str = "llama3:8b"
    openie_api_url: str = "http://localhost:11434/api/generate"  # Use native Ollama endpoint
    
    # Embedding Configuration
    embedding_model_name: str = "facebook/contriever"
    
    # Directory Configuration - use host paths
    save_dir: str = "outputs"
    
    # Performance Configuration
    embedding_batch_size: int = 32
    max_workers: int = 3  # Reduced for better stability
    
    # Generation Parameters
    temperature: float = 0.0
    max_new_tokens: int = 256
    
    # Retrieval Configuration
    retrieval_top_k: int = 200
    qa_top_k: int = 5
    linking_top_k: int = 5
    
    # Graph Configuration
    damping: float = 0.5
    passage_node_weight: float = 0.05
    synonymy_edge_sim_threshold: float = 0.8
    
    # Processing Configuration
    force_index_from_scratch: bool = False
    save_openie: bool = True
    is_directed_graph: bool = False

# Create default host config instance
host_config = HostConfig() 