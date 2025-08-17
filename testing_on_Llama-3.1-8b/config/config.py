"""
Cấu hình chung cho hệ thống HippoRAG - Enhanced Version
Compatible với HippoRAG BaseConfig với 50+ parameters
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Union, Optional, List, Dict, Any
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Enhanced Configuration System tương thích với HippoRAG BaseConfig.
    Bao gồm tất cả parameters từ HippoRAG gốc + extensions.
    """
    
    # ===== Basic Setup =====
    PROJECT_ROOT: str = field(default_factory=lambda: "/app" if os.getenv("DOCKER_ENV", "false").lower() == "true" else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ===== LLM Specific Attributes =====
    llm_name: str = field(
        default_factory=lambda: os.getenv("LLM_NAME", "llama3:8b"),
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("API_URL", "http://localhost:1234/v1/completions"),
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: {"type": "json_object"},
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    # ===== Embedding Specific Attributes =====
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "facebook/contriever"),
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_vector_dimension: int = field(
        default=768,
        metadata={"help": "Embedding vector dimension."}
    )
    
    # ===== Storage specific attributes =====
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "Force rebuilding index from scratch."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "Whether to save OpenIE results."}
    )
    
    # ===== Graph Construction Specific Attributes =====
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "Top-k for synonymy edge KNN retrieval."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Query batch size for synonymy edge KNN retrieval."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Key batch size for synonymy edge KNN retrieval."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    # ===== Retrieval Specific Attributes =====
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for passage nodes in graph."}
    )
    
    # ===== QA Specific Attributes =====
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    
    # ===== Dataset Running Specific Attributes =====
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa', 'course_data']] = field(
        default='course_data',
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets."}
    )
    
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 
        'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
        'ours'  # Full HippoRAG implementation
    ] = field(
        default="ours",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    
    # ===== Information Extraction =====
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    
    # ===== File Paths =====
    input_file: str = field(
        default_factory=lambda: os.getenv("INPUT_FILE", "triples_output_hipporag2.json"),
        metadata={"help": "Input file with extracted triples."}
    )
    output_file: str = field(
        default_factory=lambda: os.getenv("OUTPUT_FILE", "knowledge_graph_weighted.pickle"),
        metadata={"help": "Output file for knowledge graph."}
    )
    
    # ===== Directory Structure =====
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups."}
    )
    
    embedding_store_dir: str = field(
        default="embedding_stores",
        metadata={"help": "Directory for embedding stores."}
    )
    
    # ===== Legacy Parameters for Backward Compatibility =====
    API_URL: str = field(default="")
    MODEL_NAME: str = field(default="")
    EMBEDDING_MODEL: str = field(default="")
    EMBEDDING_BATCH_SIZE: int = field(default=64)
    SAVE_DIR: str = field(default="")
    LOGS_DIR: str = field(default="")
    EMBEDDING_STORE_DIR: str = field(default="")
    GRAPH_DIR: str = field(default="")
    INPUT_FILE: str = field(default="")
    OUTPUT_FILE: str = field(default="")
    DAMPING_FACTOR: float = field(default=0.5)
    LINK_TOP_K: int = field(default=5)
    DEFAULT_TOP_K: int = field(default=5)
    TOP_K: int = field(default=5)
    SYNONYMY_THRESHOLD: float = field(default=0.85)
    HIGH_SIMILARITY_THRESHOLD: float = field(default=0.98)
    MAX_WORKERS: int = field(default=3)
    CACHE_EMBEDDINGS: bool = field(default=True)
    IS_DIRECTED_GRAPH: bool = field(default=False)
    LOG_LEVEL: str = field(default="INFO")
    LOG_FORMAT: str = field(default="%(asctime)s %(levelname)s %(name)s: %(message)s")
    
    def __post_init__(self):
        """Post-initialization setup như HippoRAG BaseConfig."""
        # Setup save_dir logic như HippoRAG
        if self.save_dir is None:
            if self.dataset is None or self.dataset == 'course_data':
                self.save_dir = 'outputs'
            else:
                self.save_dir = os.path.join('outputs', self.dataset)
        
        # Check if running in Docker environment
        is_docker = os.getenv("DOCKER_ENV", "false").lower() == "true"
        
        if is_docker:
            # Use Docker paths consistent with docker_config.py
            self.SAVE_DIR = "/app/outputs"
            self.LOGS_DIR = "/app/logs"
            self.EMBEDDING_STORE_DIR = "/app/embedding_stores"
            self.GRAPH_DIR = "/app/graphs"
        else:
            # Use local development paths
            self.SAVE_DIR = os.path.join(self.PROJECT_ROOT, self.save_dir)
            self.LOGS_DIR = os.path.join(self.SAVE_DIR, "logs")
            self.EMBEDDING_STORE_DIR = self.embedding_store_dir
            self.GRAPH_DIR = os.path.join(self.SAVE_DIR, "graphs")
        
        # Sync new and legacy parameters
        self.API_URL = self.llm_base_url
        self.MODEL_NAME = self.llm_name
        self.EMBEDDING_MODEL = self.embedding_model_name
        self.EMBEDDING_BATCH_SIZE = self.embedding_batch_size
        self.INPUT_FILE = self.input_file
        self.OUTPUT_FILE = self.output_file
        self.DAMPING_FACTOR = self.damping
        self.LINK_TOP_K = self.linking_top_k
        self.DEFAULT_TOP_K = self.qa_top_k
        self.TOP_K = self.qa_top_k
        self.IS_DIRECTED_GRAPH = self.is_directed_graph
        
        logger.debug(f"Initializing save_dir to: {self.save_dir}")
        
        # Initialize directories
        self.initialize_directories()
    
    def initialize_directories(self):
        """Tạo các thư mục cần thiết nếu chưa tồn tại."""
        directories = [
            self.SAVE_DIR,
            self.LOGS_DIR,
            self.EMBEDDING_STORE_DIR,
            self.GRAPH_DIR,
            os.path.dirname(self.OUTPUT_FILE) if os.path.dirname(self.OUTPUT_FILE) else "."
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Initialized directories: {', '.join(directories)}")
        return True
    
    def get_embedding_store_path(self, name):
        """Lấy đường dẫn đến file lưu trữ embedding."""
        return os.path.join(self.EMBEDDING_STORE_DIR, f"{name}.parquet")
    
    def get_working_dir(self) -> str:
        """Get working directory based on model names như HippoRAG."""
        llm_label = self.llm_name.replace("/", "_").replace(":", "_")
        embedding_label = self.embedding_model_name.replace("/", "_").replace(":", "_")
        return os.path.join(self.save_dir, f"{llm_label}_{embedding_label}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_config(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load_config(cls, path: str):
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

# Create global config instance for backward compatibility
config = Config()

# ===== Predefined Configuration Functions =====
def get_local_config():
    """Get configuration for local deployment."""
    return Config(
        llm_name="llama3:8b",
        llm_base_url="http://localhost:11434/api/generate",
        embedding_model_name="facebook/contriever",
        openie_mode="online",
        graph_type="ours"
    )

def get_openai_config():
    """Get configuration for OpenAI."""
    return Config(
        llm_name="gpt-4o-mini",
        llm_base_url=None,  # Use OpenAI service
        embedding_model_name="text-embedding-3-large",
        openie_mode="online",
        graph_type="ours"
    )

def get_evaluation_config():
    """Get configuration optimized for evaluation."""
    return Config(
        llm_name="llama3:8b",
        llm_base_url="http://localhost:11434/api/generate",
        embedding_model_name="facebook/contriever",
        graph_type="ours",
        retrieval_top_k=200,
        qa_top_k=5,
        linking_top_k=5,
        temperature=0.0  # Deterministic for evaluation
    )

if __name__ == "__main__":
    # Test configuration
    print("=== Enhanced HippoRAG Configuration ===")
    print(f"LLM: {config.llm_name}")
    print(f"Embedding Model: {config.embedding_model_name}")
    print(f"Graph Type: {config.graph_type}")
    print(f"Working Dir: {config.get_working_dir()}")
    
    # Test save/load
    config.save_config("test_config.json")
    loaded = Config.load_config("test_config.json")
    print(f"Config loaded successfully: {loaded.llm_name == config.llm_name}") 