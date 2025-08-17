"""
Docker Configuration for HippoRAG
Simplified configuration specifically for Docker deployment
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class DockerConfig:
    """Simplified configuration for Docker environment"""
    
    # ===== Docker Environment =====
    docker_env: bool = field(default_factory=lambda: os.getenv("DOCKER_ENV", "false").lower() == "true")
    
    # ===== LLM Configuration =====
    llm_name: str = field(default_factory=lambda: os.getenv("LLM_NAME", "llama3:8b"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/api/generate"))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "2048")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0")))
    
    # ===== Embedding Configuration =====
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "facebook/contriever"))
    embedding_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "64")))
    embedding_vector_dimension: int = field(default=768)  # 🔥 MISSING ATTRIBUTE ADDED
    
    # ===== Directories (Docker paths) =====
    save_dir: str = field(default_factory=lambda: os.getenv("SAVE_DIR", "/app/outputs"))
    logs_dir: str = field(default_factory=lambda: os.getenv("LOGS_DIR", "/app/logs"))
    embedding_store_dir: str = field(default_factory=lambda: os.getenv("EMBEDDING_STORE_DIR", "/app/embedding_stores"))
    graph_dir: str = field(default="/app/graphs")
    
    # ===== Storage Configuration =====
    storage_type: str = field(default_factory=lambda: os.getenv("STORAGE_TYPE", "file-based"))
    
    # ===== Performance Configuration =====
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "3")))
    cache_embeddings: bool = field(default_factory=lambda: os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true")
    
    # ===== Dataset Configuration =====
    dataset: str = field(default="course_data")
    graph_type: str = field(default="ours")
    
    # ===== Processing Configuration =====
    force_index_from_scratch: bool = field(default=False)
    save_openie: bool = field(default=True)
    is_directed_graph: bool = field(default=False)
    openie_mode: str = field(default="online")
    
    # ===== Retrieval Configuration =====
    retrieval_top_k: int = field(default=200)
    qa_top_k: int = field(default=5)
    linking_top_k: int = field(default=5)
    
    # ===== Graph Configuration =====
    damping: float = field(default=0.5)
    passage_node_weight: float = field(default=0.05)
    synonymy_edge_sim_threshold: float = field(default=0.8)  # Changed from 0.85 to 0.8 to match HippoRAG
    synonymy_edge_topk: int = field(default=2047)
    synonymy_edge_query_batch_size: int = field(default=1000)
    synonymy_edge_key_batch_size: int = field(default=10000)
    
    def __post_init__(self):
        """Initialize Docker-specific settings"""
        # Ensure directories exist
        self.initialize_directories()
        
        # Set up logging
        self.setup_logging()
        
        logger.info(f"Docker configuration initialized")
        logger.info(f"LLM: {self.llm_name}")
        logger.info(f"Embedding Model: {self.embedding_model_name}")
        logger.info(f"Storage Type: {self.storage_type}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Save Directory: {self.save_dir}")
    
    def initialize_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.save_dir,
            self.logs_dir,
            self.embedding_store_dir,
            self.graph_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_file = os.path.join(self.logs_dir, "hipporag.log")
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama-specific configuration"""
        return {
            "base_url": self.llm_base_url,
            "model": self.llm_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding-specific configuration"""
        return {
            "model_name": self.embedding_model_name,
            "batch_size": self.embedding_batch_size,
            "store_dir": self.embedding_store_dir
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return {
            "type": self.storage_type,
            "embedding_store_dir": self.embedding_store_dir,
            "save_dir": self.save_dir,
            "graph_dir": self.graph_dir
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)

# Create global docker_config instance
docker_config = DockerConfig()
