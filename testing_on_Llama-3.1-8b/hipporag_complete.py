import json
import os
import logging
import time
import numpy as np
import igraph as ig
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import requests
import re
import ast
import difflib
from copy import deepcopy
from pydantic import BaseModel, Field

# Import các modules đã có
from utils.embedding_store import EmbeddingStore
from utils.simple_embedding_model import SimpleEmbeddingModel
from hipporag_openie import SimpleOpenIE
from hipporag_prompts import PromptTemplateManager
from utils.hipporag_utils import (
    compute_mdhash_id, 
    reformat_openie_results, 
    extract_entity_nodes, 
    flatten_facts, 
    text_processing,
    min_max_normalize
)
from utils.embed_utils import retrieve_knn
from config.config import Config, config

logger = logging.getLogger(__name__)

@dataclass
class BaseConfig(Config):
    """Configuration class kế thừa từ Config trong config.py"""
    pass

# Tạo alias để tương thích
HippoRAGConfig = BaseConfig

@dataclass
class QuerySolution:
    """
    Lớp lưu trữ kết quả truy vấn
    """
    question: str
    docs: List[str]
    doc_scores: List[float]
    answer: Optional[str] = None
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None

class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")

class DSPyFilter:
    """
    DSPy Filter using PromptTemplateManager
    """
    
    def __init__(self, hipporag):
        self.hipporag = hipporag
        self.model_name = hipporag.global_config.llm_name
        self.default_gen_kwargs = {'max_completion_tokens': 1024}

        # Use PromptTemplateManager instead of hardcoded prompts
        self.prompt_manager = PromptTemplateManager()

    def parse_filter(self, response):
        """
        Parse response từ LLM với improved parsing
        Input: response (str) - LLM response text
        Output: List - parsed facts
        Chức năng: Parse LLM response để extract facts
        """
        logger.info(f"PARSE_FILTER DEBUG - Input response length: {len(response)}")
        logger.info(f"Raw response:\n{response}")
        
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        
        logger.info(f"Found {len(sections)} sections: {[k for k, v in sections]}")
        
        for k, value in sections:
            if k == "fact_after_filter":
                logger.info(f"Processing fact_after_filter section: {value}")
                try:
                    try:
                        parsed_value = json.loads(value)
                        logger.info(f"JSON parsing successful: {parsed_value}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed: {e}")
                        try:
                            parsed_value = ast.literal_eval(value)
                            logger.info(f"AST parsing successful: {parsed_value}")
                        except (ValueError, SyntaxError) as e2:
                            logger.warning(f"AST parsing failed: {e2}")
                            parsed_value = value
                    
                    # Extract facts from parsed value
                    if isinstance(parsed_value, dict) and 'fact' in parsed_value:
                        parsed = parsed_value['fact']
                        logger.info(f"Standard parsing successful: {len(parsed)} facts")
                    elif isinstance(parsed_value, list):
                        parsed = parsed_value
                        logger.info(f"Direct list parsing successful: {len(parsed)} facts")
                        
                except Exception as e:
                    logger.error(f"Error parsing field {k}: {e}")
                    parsed = []

        # Nếu standard parsing thất bại, thử alternative JSON parsing
        if not parsed:
            logger.warning("Standard parsing failed, trying alternative JSON parsing...")
            try:
                # Tìm JSON pattern trong response
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    logger.info(f"Found JSON pattern: {json_str}")
                    
                    json_data = json.loads(json_str)
                    logger.info(f"Parsed JSON data: {json_data}")
                    
                    if 'fact' in json_data and isinstance(json_data['fact'], list):
                        parsed = json_data['fact']
                        logger.info(f"Alternative JSON parsing successful: found {len(parsed)} facts")
                        logger.info(f"Facts: {parsed}")
                    else:
                        logger.warning(f"JSON found but no 'fact' key or wrong format")
                        logger.warning(f"JSON keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
                else:
                    logger.warning("No JSON pattern found in response")
                    
            except Exception as e:
                logger.error(f"Alternative JSON parsing failed: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"FINAL RESULT: Parsed {len(parsed)} facts from filter response")
        return parsed

    def llm_call(self, question, fact_before_filter):
        """
        Gọi LLM using PromptTemplateManager
        Input: question (str), fact_before_filter (dict)
        Output: str - LLM response
        Chức năng: Generate prompt và call LLM API
        """
        # Use PromptTemplateManager to generate messages
        messages = self.prompt_manager.render('dspy_filter', 
                question=question, 
                                            fact_before_filter=fact_before_filter)

        try:
            # Convert messages to prompt format for our API
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

            # Use native Ollama API format
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.hipporag.global_config.temperature,
                    "num_predict": 512,
                    "top_p": 0.9
                }
            }

            response = requests.post(self.hipporag.global_config.llm_base_url, json=payload, timeout=120)
            response.raise_for_status()
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.warning(f"LLM API request failed with status code: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return ""

    def rerank(self, query: str, candidate_items: List[Tuple], candidate_indices: List[int], len_after_rerank: int = None) -> Tuple[List[int], List[Tuple], dict]:
        """
        Rerank với fallback strategy
        Input: query (str), candidate_items (List[Tuple]), candidate_indices (List[int]), len_after_rerank (int)
        Output: (List[int], List[Tuple], dict) - reranked indices, items, metadata
        Chức năng: Rerank candidate items theo query relevance
        """
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        
        try:
            response = self.llm_call(query, json.dumps(fact_before_filter))
            if response.strip():
                generated_facts = self.parse_filter(response)
            else:
                generated_facts = []
        except Exception as e:
            logger.warning(f'DSPy rerank exception: {e}')
            generated_facts = []

        # Process generated facts to match with candidates
        result_indices = []
        for generated_fact in generated_facts:
            try:
                # Find closest match using difflib như HippoRAG gốc
                closest_matched_fact = difflib.get_close_matches(
                    str(generated_fact), 
                    [str(candidate_item) for candidate_item in candidate_items], 
                    n=1, 
                    cutoff=0.0
                )
                
                if closest_matched_fact:
                    # Find the index of the matched fact
                    matched_fact_str = closest_matched_fact[0]
                    for i, candidate_item in enumerate(candidate_items):
                        if str(candidate_item) == matched_fact_str:
                            result_indices.append(i)
                            break
                            
            except Exception as e:
                logger.warning(f'result_indices exception: {e}')

        # If no valid matches found, return original order (fallback)
        if not result_indices:
            logger.info("No valid matches found in reranking, using original order")
            result_indices = list(range(len(candidate_items)))

        # Apply length limit
        if len_after_rerank:
            result_indices = result_indices[:len_after_rerank]

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        
        return sorted_candidate_indices, sorted_candidate_items, {'confidence': 'llm' if generated_facts else 'fallback'}

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

class HippoRAG:
    """
    FIXED: HippoRAG implementation - KHÔNG CHUNKING
    """
    
    def __init__(self, global_config=None, save_dir=None, llm_model_name=None, embedding_model_name=None, llm_base_url=None):
        # Initialize configuration
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
            
        # Override with explicit parameters if provided
        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name
        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name
        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url
            
        # Set save directory
        if save_dir is None:
            self.save_dir = os.path.join(os.getcwd(), 'outputs')
        else:
            self.save_dir = save_dir
        
        # Validate save directory path
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Cannot create save_dir {self.save_dir}: {e}")
            self.save_dir = os.path.join(os.getcwd(), 'outputs')
            os.makedirs(self.save_dir, exist_ok=True)
            
        # Log configuration
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"LLM: {self.global_config.llm_name}")
        logger.info(f"Embedding: {self.global_config.embedding_model_name}")
        
        # Store filepaths
        self.chunk_db_path = os.path.join(self.save_dir, 'chunks')
        self.entity_db_path = os.path.join(self.save_dir, 'entities')
        self.fact_db_path = os.path.join(self.save_dir, 'facts')
        self.openie_results_path = os.path.join(self.save_dir, 'openie_results.json')
        self._graph_pickle_filename = os.path.join(self.save_dir, 'graph.pkl')

        # THÊM QUERY CACHING SYSTEM
        self.query_to_embedding = {'triple': {}, 'passage': {}}
        
        # Timing variables
        self.last_query_timing = {}  # Track timing for different components
        
        # Initialize embedding model
        self.embedding_model = SimpleEmbeddingModel(
            model_name=self.global_config.embedding_model_name
        )
        
        # Initialize embedding stores
        self.chunk_embedding_store = EmbeddingStore(
            embedding_model=self.embedding_model,
            db_filename=self.chunk_db_path,
            batch_size=self.global_config.embedding_batch_size,
            namespace='chunk'
        )
        
        self.entity_embedding_store = EmbeddingStore(
            embedding_model=self.embedding_model,
            db_filename=self.entity_db_path,
            batch_size=self.global_config.embedding_batch_size,
            namespace='entity'
        )
        
        self.fact_embedding_store = EmbeddingStore(
            embedding_model=self.embedding_model,
            db_filename=self.fact_db_path,
            batch_size=self.global_config.embedding_batch_size,
            namespace='fact'
        )
        
        # Initialize OpenIE
        self.openie = SimpleOpenIE(
            api_url=self.global_config.llm_base_url,
            model_name=self.global_config.llm_name
        )
        
        # Initialize prompt template manager
        self.prompt_manager = PromptTemplateManager()
        
        # Initialize rerank filter
        self.rerank_filter = DSPyFilter(self)
        
        # Initialize graph
        self.graph = self.initialize_graph()
        
        # Initialize retrieval state
        self.ready_to_retrieve = False
        
        # Initialize graph construction variables
        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}
        
        # Initialize retrieval variables
        self.node_name_to_vertex_idx = {}
        self.passage_node_keys = []
        self.passage_node_idxs = []
        self.entity_node_keys = []
        self.fact_node_keys = []
        self.entity_embeddings = None
        self.passage_embeddings = None
        self.fact_embeddings = None

    def initialize_graph(self):
        """Initialize graph từ pickle hoặc tạo mới theo đúng HippoRAG gốc"""
        preloaded_graph = None
        
        # FIX: Clear all stores when force_index_from_scratch is True
        if self.global_config.force_index_from_scratch:
            logger.info("Force index from scratch: clearing all existing data...")
            
            # Clear graph pickle
            if os.path.exists(self._graph_pickle_filename):
                os.remove(self._graph_pickle_filename)
                logger.info(f"Removed existing graph pickle: {self._graph_pickle_filename}")
                
            # Clear embedding stores
            for store_path in [self.chunk_db_path, self.entity_db_path, self.fact_db_path]:
                if os.path.exists(store_path):
                    import shutil
                    shutil.rmtree(store_path)
                    logger.info(f"Removed existing store: {store_path}")
                    
            # Clear OpenIE results
            if os.path.exists(self.openie_results_path):
                os.remove(self.openie_results_path)
                logger.info(f"Removed existing OpenIE results: {self.openie_results_path}")
                
        elif os.path.exists(self._graph_pickle_filename):
            try:
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)
                logger.info(f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges")
            except Exception as e:
                logger.warning(f"Failed to load graph: {e}")
                preloaded_graph = None
        
        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            # When loading existing graph, regenerate metadata
            self._regenerate_graph_metadata(preloaded_graph)
            return preloaded_graph

    def _regenerate_graph_metadata(self, graph):
        """Regenerate node_to_node_stats and ent_node_to_chunk_ids from existing graph"""
        try:
            logger.info("Regenerating graph metadata from existing graph structure...")
            
            # Reset metadata
            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            
            # Regenerate node_to_node_stats from edges
            for edge in graph.es:
                source_name = graph.vs[edge.source]['name'] if 'name' in graph.vs[edge.source].attributes() else None
                target_name = graph.vs[edge.target]['name'] if 'name' in graph.vs[edge.target].attributes() else None
                weight = edge.get('weight', 1.0) if 'weight' in edge.attributes() else 1.0
                
                if source_name and target_name:
                    self.node_to_node_stats[(source_name, target_name)] = weight
                    # Add reverse edge for undirected graph
                    if not self.global_config.is_directed_graph:
                        self.node_to_node_stats[(target_name, source_name)] = weight
            
            # Regenerate ent_node_to_chunk_ids from graph structure
            for vertex in graph.vs:
                if 'name' not in vertex.attributes():
                    continue
                    
                node_name = vertex['name']
                
                # If it's an entity node, find connected chunk nodes
                if node_name.startswith('entity-'):
                    connected_chunks = set()
                    
                    # Get neighbors
                    neighbors = graph.neighbors(vertex.index)
                    for neighbor_idx in neighbors:
                        neighbor = graph.vs[neighbor_idx]
                        if 'name' in neighbor.attributes():
                            neighbor_name = neighbor['name']
                            if neighbor_name.startswith('chunk-'):
                                connected_chunks.add(neighbor_name)
                    
                    if connected_chunks:
                        self.ent_node_to_chunk_ids[node_name] = connected_chunks
            
            logger.info(f"Regenerated metadata: {len(self.node_to_node_stats)} node relationships, {len(self.ent_node_to_chunk_ids)} entity-chunk mappings")
            
        except Exception as e:
            logger.warning(f"Error regenerating graph metadata: {e}")
            # Keep empty dictionaries as fallback
            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}

    def index(self, docs: List[str]):
        """Index documents theo workflow của HippoRAG gốc"""
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE")
        
        # 1. Insert chunks vào embedding store theo đúng HippoRAG gốc
        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()
        
        # 2. Load existing OpenIE results
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}
        
        # 3. Perform OpenIE if needed
        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)
        
        # 4. Save OpenIE results
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)
        
        # 5. Reformat OpenIE results
        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)
        
        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)
        
        # 6. Prepare data_store theo đúng HippoRAG gốc
        chunk_ids = list(chunk_to_rows.keys())
        
        # CRITICAL: Apply text_processing (includes .lower()) to triples như HippoRAG gốc
        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)
        
        # 7. Encode entities and facts theo đúng HippoRAG gốc
        try:
            logger.info(f"Encoding Entities")
            # THEO ĐÚNG HIPPORAG GỐC: Entity nodes đã được processed với .lower() trong text_processing
            # Entity embedding store đã có namespace='entity' nên sẽ tự động tạo prefix "entity-"
            self.entity_embedding_store.insert_strings(entity_nodes)
            
            logger.info(f"Encoding Facts")
            self.fact_embedding_store.insert_strings([str(fact) for fact in facts])
        except Exception as e:
            logger.error(f"Error encoding entities/facts: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 8. Construct graph theo đúng HippoRAG gốc
        try:
            logger.info(f"Constructing Graph")
            logger.info(f"DEBUG: chunk_ids: {len(chunk_ids)}, chunk_triples: {len(chunk_triples)}, facts: {len(facts)}")
            
            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            
            logger.info("DEBUG: Adding fact edges...")
            self.add_fact_edges(chunk_ids, chunk_triples)
            logger.info(f"DEBUG: node_to_node_stats after fact edges: {len(self.node_to_node_stats)}")
            
            logger.info("DEBUG: Adding passage edges...")
            num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)
            logger.info(f"DEBUG: num_new_chunks: {num_new_chunks}, ent_node_to_chunk_ids: {len(self.ent_node_to_chunk_ids)}")
            
            # ALWAYS build graph if we have stats, regardless of new chunks
            if len(self.node_to_node_stats) > 0 or len(self.ent_node_to_chunk_ids) > 0:
                logger.info(f"Found {num_new_chunks} new chunks and {len(self.node_to_node_stats)} node relationships to save into graph.")
                
                logger.info("DEBUG: Adding synonymy edges...")
                self.add_synonymy_edges()
                logger.info(f"DEBUG: node_to_node_stats after synonymy: {len(self.node_to_node_stats)}")
                
                logger.info("DEBUG: Augmenting graph...")
                self.augment_graph()
                logger.info(f"DEBUG: Graph after augment: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
                
                logger.info("DEBUG: Saving graph...")
                self.save_igraph()
            else:
                logger.warning("DEBUG: No relationships found, skipping graph augmentation")
            
            logger.info("Indexing completed successfully!")
        except Exception as e:
            logger.error(f"Error constructing graph: {e}")
            import traceback
            traceback.print_exc()
            raise

    # THÊM: augment_graph method theo đúng HippoRAG gốc
    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """
        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        
        # THÊM: Detailed logging với các thống kê
        graph_info = self.get_graph_info()
        
        logger.info("DETAILED GRAPH STATISTICS:")
        logger.info(f"   # of Unique Nodes (N): {graph_info.get('unique_nodes_N', 'N/A')}")
        logger.info(f"   # of Unique Edges (E): {graph_info.get('unique_edges_E', 'N/A')}")
        logger.info(f"   # of Unique Triples: {graph_info.get('unique_triples', 'N/A')}")
        logger.info(f"   # of Synonym Edges (E'): {graph_info.get('synonym_edges_E_prime', 'N/A')}")
        logger.info("=" * 60)
        
        print(graph_info)

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """Load existing OpenIE results theo đúng HippoRAG gốc"""
        chunk_keys_to_save = set()
        
        if os.path.isfile(self.openie_results_path):
            try:
                with open(self.openie_results_path, 'r', encoding='utf-8') as f:
                    openie_results = json.load(f)
                all_openie_info = openie_results.get('docs', [])
                
                # Standardize indices theo đúng HippoRAG gốc
                renamed_openie_info = []
                for openie_info in all_openie_info:
                    if 'idx' not in openie_info or not openie_info['idx']:
                        openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                    renamed_openie_info.append(openie_info)
                
                all_openie_info = renamed_openie_info
                existing_openie_keys = set([info['idx'] for info in all_openie_info])
                
                for chunk_key in chunk_keys:
                    if chunk_key not in existing_openie_keys:
                        chunk_keys_to_save.add(chunk_key)
                        
                logger.info(f"Loaded {len(all_openie_info)} existing OpenIE results, {len(chunk_keys_to_save)} new chunks to process")
                
            except Exception as e:
                logger.error(f"Error loading OpenIE results: {e}")
                all_openie_info = []
                chunk_keys_to_save = set(chunk_keys)
        else:
            logger.info("No existing OpenIE results file found")
            all_openie_info = []
            chunk_keys_to_save = set(chunk_keys)
        
        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self, all_openie_info: List[dict], chunks_to_save: Dict[str, dict], 
                           ner_results_dict: Dict, triple_results_dict: Dict) -> List[dict]:
        """Merge OpenIE results theo đúng HippoRAG gốc"""
        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {
                'idx': chunk_key, 
                'passage': passage,
                'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                'extracted_triples': triple_results_dict[chunk_key].triples
            }
            all_openie_info.append(chunk_openie_info)
        
        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """Save OpenIE results với comprehensive stats"""
        if len(all_openie_info) == 0:
            logger.warning("No OpenIE results to save")
            return
            
        try:
            # Basic entity stats
            sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk.get('extracted_entities', [])])
            sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk.get('extracted_entities', [])])
            num_phrases = sum([len(chunk.get('extracted_entities', [])) for chunk in all_openie_info])
            
            # Triple stats  
            total_triples = sum([len(chunk.get('extracted_triples', [])) for chunk in all_openie_info])
            docs_with_entities = sum([1 for chunk in all_openie_info if len(chunk.get('extracted_entities', [])) > 0])
            docs_with_triples = sum([1 for chunk in all_openie_info if len(chunk.get('extracted_triples', [])) > 0])
            
            # Document stats
            total_docs = len(all_openie_info)
            doc_lengths = [len(chunk.get('passage', '')) for chunk in all_openie_info]
            
            # Compute averages
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
            
            # Enhanced stats object
            openie_stats = {
                # Document stats
                'total_documents': total_docs,
                'avg_doc_length': round(sum(doc_lengths) / len(doc_lengths), 1) if doc_lengths else 0,
                'min_doc_length': min(doc_lengths) if doc_lengths else 0,
                'max_doc_length': max(doc_lengths) if doc_lengths else 0,
                
                # Entity stats
                'total_entities': num_phrases,
                'avg_entities_per_doc': round(num_phrases / total_docs, 2) if total_docs > 0 else 0,
                'docs_with_entities': docs_with_entities,
                'entity_success_rate': round(docs_with_entities / total_docs * 100, 1) if total_docs > 0 else 0,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words,
                
                # Triple stats
                'total_triples': total_triples,
                'avg_triples_per_doc': round(total_triples / total_docs, 2) if total_docs > 0 else 0,
                'docs_with_triples': docs_with_triples,
                'triple_success_rate': round(docs_with_triples / total_docs * 100, 1) if total_docs > 0 else 0,
                
                # Quality metrics
                'overall_success_rate': round((docs_with_entities + docs_with_triples) / (2 * total_docs) * 100, 1) if total_docs > 0 else 0
            }
            
            openie_dict = {
                'docs': all_openie_info,
                'extraction_stats': openie_stats,
                # Legacy fields for compatibility
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.openie_results_path), exist_ok=True)
            
            with open(self.openie_results_path, 'w', encoding='utf-8') as f:
                json.dump(openie_dict, f, ensure_ascii=False, indent=2)
            
            # Enhanced logging
            logger.info(f"OpenIE results saved to {self.openie_results_path}")
            logger.info(f"FINAL EXTRACTION STATS:")
            logger.info(f"   Documents: {openie_stats['total_documents']}")
            logger.info(f"   Entities: {openie_stats['total_entities']} (avg: {openie_stats['avg_entities_per_doc']}/doc)")
            logger.info(f"   Triples: {openie_stats['total_triples']} (avg: {openie_stats['avg_triples_per_doc']}/doc)")
            logger.info(f"   Success rates: Entities {openie_stats['entity_success_rate']}%, Triples {openie_stats['triple_success_rate']}%")
            
        except Exception as e:
            logger.error(f"Error saving OpenIE results: {e}")
            raise

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[List[List[str]]]):
        """Add fact edges theo đúng HippoRAG gốc"""
        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """Add passage edges theo đúng HippoRAG gốc"""
        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.
        """
        logger.info(f"Expanding graph with synonymy edges")

        entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

        logger.info(f"Added {num_synonym_triple} synonym edges to graph")

    def add_new_nodes(self):
        """
        Add new nodes với duplicate tracking
        Input: None (uses instance variables)
        Output: None (modifies graph in-place)
        Chức năng: Add unique nodes to graph while tracking duplicates
        """
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}
        logger.info(f"DEBUG: Graph currently has {len(existing_nodes)} existing nodes")

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()
        logger.info(f"DEBUG: Entity store has {len(entity_to_row)} entities, Passage store has {len(passage_to_row)} passages")
        
        # Handle entity nodes
        entity_ids = list(entity_to_row.keys())
        logger.info(f"DEBUG: Sample entity IDs: {entity_ids[:5] if entity_ids else []}")
        entity_with_prefix = [eid for eid in entity_ids if eid.startswith('entity-')]
        logger.info(f"DEBUG: Entities with entity- prefix: {len(entity_with_prefix)}/{len(entity_to_row)}")
        
        node_to_rows = {}
        node_to_rows.update(entity_to_row)
        node_to_rows.update(passage_to_row)

        # THÊM: Duplicate tracking
        total_candidate_nodes = len(node_to_rows)
        duplicate_nodes = []
        merged_nodes = []
        new_node_ids = []
        
        for node_id in node_to_rows:
            if node_id in existing_nodes:
                duplicate_nodes.append(node_id)
            else:
                new_node_ids.append(node_id)
                
        # Check for potential merged entities (same ID, different sources)
        for node_id in node_to_rows:
            if node_id in entity_to_row and node_id in passage_to_row:
                merged_nodes.append(node_id)

        # THÊM: Detailed duplicate logging
        num_duplicates = len(duplicate_nodes)
        num_merged = len(merged_nodes)
        num_new = len(new_node_ids)
        
        # Node duplicate analysis
        logger.info(f"NODE DUPLICATE ANALYSIS:")
        logger.info(f"   Total candidate nodes: {total_candidate_nodes}")
        logger.info(f"   New nodes to add: {num_new}")
        logger.info(f"   Duplicate nodes (skipped): {num_duplicates}")
        logger.info(f"   Deduplication rate: {(num_duplicates/total_candidate_nodes*100):.1f}%")
        
        # Show sample duplicates for debugging
        sample_duplicates = duplicate_nodes[:3] if duplicate_nodes else []
        logger.info(f"   Sample duplicate nodes: {sample_duplicates}")
        
        # Add new nodes to graph
        if new_node_ids:
            # Add nodes to graph
            try:
                # Add vertices first
                start_index = self.graph.vcount()
                self.graph.add_vertices(len(new_node_ids))
                
                # Set node attributes after adding
                for i, node_id in enumerate(new_node_ids):
                    self.graph.vs[start_index + i]["name"] = node_id
                
                logger.info(f"DEBUG: Successfully added {num_new} new nodes to graph")
                logger.info(f"DEBUG: Graph now has {self.graph.vcount()} total nodes")
            except Exception as e:
                logger.error(f"Error adding nodes to graph: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info(f"DEBUG: No new nodes to add")

    def _should_merge_node(self, existing_node, new_node: dict) -> bool:
        """Check if nodes should be merged based on content differences"""
        # Simple heuristic: merge if content is different
        # existing_node is igraph.Vertex object, new_node is dict
        existing_content = existing_node.get('content', '') if hasattr(existing_node, 'get') else existing_node.attributes().get('content', '')
        new_content = new_node.get('content', '')
        
        # If content is different, we might want to merge/update
        return existing_content != new_content

    def add_new_edges(self):
        """
        Add new edges theo đúng HippoRAG
        Input: None (uses instance variables)
        Output: None (modifies graph in-place)
        Chức năng: Add edges between nodes based on relationships
        """
        from collections import defaultdict
        
        # Get edge sources from various relationships
        edge_source_node_keys = []
        
        # Add entity-entity edges
        edge_source_node_keys.extend(list(self.node_to_node_stats.keys()))
        
        # Add entity-chunk edges
        for entity_node in self.ent_node_to_chunk_ids:
            for chunk_id in self.ent_node_to_chunk_ids[entity_node]:
                edge_source_node_keys.append((entity_node, chunk_id))
        
        logger.info(f"DEBUG: Processing {len(edge_source_node_keys)} potential edges")
        
        # Get current node mapping
        current_node_ids = {v["name"]: i for i, v in enumerate(self.graph.vs) if "name" in v.attributes()}
        logger.info(f"DEBUG: Graph has {len(current_node_ids)} nodes available for edge creation")
        
        # Filter edges to only include nodes that exist in graph
        valid_edges = []  
        valid_weights = {'weight': []}
        invalid_edges = []
        
        for edge_key in edge_source_node_keys:
            if isinstance(edge_key, tuple) and len(edge_key) == 2:
                source_node, target_node = edge_key
                
                if source_node in current_node_ids and target_node in current_node_ids:
                    valid_edges.append((current_node_ids[source_node], current_node_ids[target_node]))
                    weight = self.node_to_node_stats.get((source_node, target_node), 1.0)
                    valid_weights['weight'].append(weight)
                else:
                    invalid_edges.append(edge_key)
                    
        logger.info(f"DEBUG: Found {len(valid_edges)} valid edges and {len(invalid_edges)} invalid edges")
        
        # THEO ĐÚNG HIPPORAG: Luôn gọi add_edges (không có điều kiện if valid_edges)
        if len(valid_edges) > 0:
            self.graph.add_edges(valid_edges, attributes=valid_weights)
            logger.info(f"DEBUG: Successfully added {len(valid_edges)} edges to graph")
        else:
            logger.warning(f"DEBUG: No valid edges to add to graph")

    def get_graph_info(self) -> Dict:
        """THÊM: Get graph info theo đúng HippoRAG gốc"""
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        # THÊM: Detailed statistics theo yêu cầu
        # # of Unique Nodes (N)
        if hasattr(self.graph, 'vcount'):
            graph_info["unique_nodes_N"] = self.graph.vcount()
        else:
            graph_info["unique_nodes_N"] = graph_info["num_total_nodes"]

        # # of Unique Edges (E) 
        if hasattr(self.graph, 'ecount'):
            graph_info["unique_edges_E"] = self.graph.ecount()
        else:
            graph_info["unique_edges_E"] = len(self.node_to_node_stats)

        # # of Unique Triples - same as extracted triples
        graph_info["unique_triples"] = graph_info["num_extracted_triples"]

        # # of Synonym Edges (E')
        graph_info["synonym_edges_E_prime"] = graph_info['num_synonymy_triples']

        return graph_info

    def save_igraph(self):
        """Save graph to pickle file theo đúng HippoRAG gốc"""
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Graph saved to {self._graph_pickle_filename}")

    def prepare_retrieval_objects(self):
        """Prepare objects needed for retrieval theo đúng HippoRAG gốc"""
        if self.ready_to_retrieve:
            return
        
        try:
            # Always initialize basic structures
            self.passage_node_keys = self.chunk_embedding_store.get_all_ids()
            self.entity_node_keys = self.entity_embedding_store.get_all_ids()
            self.fact_node_keys = self.fact_embedding_store.get_all_ids()
            
            # Create node mappings - only if graph has nodes
            if self.graph.vcount() == 0:
                logger.warning("Graph is empty, cannot prepare retrieval objects")
                self.node_name_to_vertex_idx = {}
                self.passage_node_idxs = []
                # Still prepare embeddings for fallback
                logger.info("Loading embeddings for fallback retrieval.")
                if self.entity_node_keys:
                    self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
                if self.passage_node_keys:
                    self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))
                if self.fact_node_keys:
                    self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))
                self.ready_to_retrieve = True
                return
            
            self.node_name_to_vertex_idx = {v['name']: v.index for v in self.graph.vs if 'name' in v.attributes()}
            
            # Get passage node indices and keys theo đúng HippoRAG gốc
            self.passage_node_keys = self.chunk_embedding_store.get_all_ids()
            self.passage_node_idxs = [self.node_name_to_vertex_idx[key] for key in self.passage_node_keys if key in self.node_name_to_vertex_idx]
            
            # Get entity node keys
            self.entity_node_keys = self.entity_embedding_store.get_all_ids()
            
            # Get fact node keys
            self.fact_node_keys = self.fact_embedding_store.get_all_ids()
            
            # Load embeddings theo đúng HippoRAG gốc
            logger.info("Loading embeddings.")
            self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
            self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))
            self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))
            
            self.ready_to_retrieve = True
            logger.info(f"Retrieval objects prepared: {len(self.passage_node_keys)} passages, {len(self.fact_node_keys)} facts")
            
        except Exception as e:
            logger.error(f"Error preparing retrieval objects: {e}")
            self.ready_to_retrieve = False
            raise

    def get_query_embeddings(self, queries: List[str]):
        """
        THÊM QUERY CACHING SYSTEM theo đúng HippoRAG gốc
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping.
        """
        all_query_strings = []
        for query in queries:
            if (query not in self.query_to_embedding['triple'] or 
                query not in self.query_to_embedding['passage']):
                all_query_strings.append(query)
        
        if len(all_query_strings) > 0:
            # Encode cho fact retrieval
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding
                
            # Encode cho passage retrieval  
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """OPTIMIZED: Get fact scores với cached query và pre-computed embeddings"""
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        
        if len(self.fact_node_keys) == 0:
            logger.warning("No facts available for scoring")
            return np.array([])
        
        try:
            # USE CACHED QUERY EMBEDDING
            query_embedding = self.query_to_embedding['triple'].get(query, None)
            if query_embedding is None:
                query_embedding = self.embedding_model.batch_encode([query])[0]
                self.query_to_embedding['triple'][query] = query_embedding
            
            # USE PRE-COMPUTED FACT EMBEDDINGS
            similarities = np.dot(self.fact_embeddings, query_embedding) / (
                np.linalg.norm(self.fact_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error getting fact scores: {e}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """OPTIMIZED: Dense passage retrieval với cached query và pre-computed embeddings"""
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        
        try:
            # USE CACHED QUERY EMBEDDING
            query_embedding = self.query_to_embedding['passage'].get(query, None)
            if query_embedding is None:
                query_embedding = self.embedding_model.batch_encode([query])[0]
                self.query_to_embedding['passage'][query] = query_embedding
            
            # USE PRE-COMPUTED PASSAGE EMBEDDINGS
            query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
            query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
            query_doc_scores = min_max_normalize(query_doc_scores)
            
            # Sort by similarity
            sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
            sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
            
            return sorted_doc_ids, sorted_doc_scores
            
        except Exception as e:
            logger.error(f"Error in dense passage retrieval: {e}")
            return np.array([]), np.array([])

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """FIXED: Rerank facts using DSPy filter theo đúng HippoRAG gốc"""
        link_top_k = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score - EXACT HippoRAG logic
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            
            # FIX: Parse fact content correctly như HippoRAG gốc
            candidate_facts = []
            for fact_id in real_candidate_fact_ids:
                try:
                    fact_content = fact_row_dict[fact_id]['content']
                    # HippoRAG gốc uses eval() to parse fact triples
                    parsed_fact = eval(fact_content)
                    candidate_facts.append(tuple(parsed_fact))
                except Exception as e:
                    logger.warning(f"Error parsing fact {fact_id}: {e}")
                    continue
            
            # Rerank the facts using DSPy filter
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}

    def get_top_k_weights(self, link_top_k: int, all_phrase_weights: np.ndarray, linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Get top k weights theo đúng HippoRAG gốc"""
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str, link_top_k: int, query_fact_scores: np.ndarray,
                                      top_k_facts: List[Tuple], top_k_fact_indices: List[str],
                                      passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """FIXED: Graph search with fact entities theo đúng HippoRAG gốc"""
        # Assigning phrase weights based on selected facts from previous steps
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        for rank, f in enumerate(top_k_facts):
            # CRITICAL: Apply .lower() to fact entities như HippoRAG gốc
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        phrase_weights[phrase_id] /= len(self.ent_node_to_chunk_ids[phrase_key])

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)

        # Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        # Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        # Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time = (ppr_end - ppr_start)  # FIX: Không accumulate, chỉ lưu timing của query cuối

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Run Personalized PageRank theo đúng HippoRAG gốc"""
        try:
            # If graph has no edges, fallback to dense retrieval
            if self.graph.ecount() == 0:
                logger.warning("Graph has no edges, falling back to dense retrieval")
                return self.dense_passage_retrieval("")
            
            if damping is None: 
                damping = 0.5  # for potential compatibility
            reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
            
            pagerank_scores = self.graph.personalized_pagerank(
                vertices=range(len(self.node_name_to_vertex_idx)),
                damping=damping,
                directed=False,
                weights='weight',
                reset=reset_prob,
                implementation='prpack'
            )

            doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
            sorted_doc_ids = np.argsort(doc_scores)[::-1]
            sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

            return sorted_doc_ids, sorted_doc_scores
            
        except Exception as e:
            logger.error(f"Error in PPR: {e}")
            # Fallback to dense retrieval
            return self.dense_passage_retrieval("")

    def retrieve(self, queries: List[str], num_to_retrieve: int = None, gold_docs: List[List[str]] = None) -> List[QuerySolution]:
        """Main retrieval method theo đúng HippoRAG gốc"""
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        
        # THÊM QUERY EMBEDDING PREPARATION
        self.get_query_embeddings(queries)
        
        retrieval_results = []
        
        # Track timing for the entire batch
        batch_start_time = time.time()
        
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            # Reset timing for this individual query
            query_start_time = time.time()
            self.ppr_time = 0
            self.rerank_time = 0
            
            # 1. Get fact scores and rerank
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()
            self.rerank_time = rerank_end - rerank_start
            
            # 2. Graph search or fallback to DPR theo đúng logic HippoRAG gốc
            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(
                    query=query,
                    link_top_k=self.global_config.linking_top_k,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    passage_node_weight=self.global_config.passage_node_weight
                )
            
            # 3. Get top documents theo đúng HippoRAG gốc
            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            
            # Calculate individual query timing
            query_end_time = time.time()
            individual_query_time = query_end_time - query_start_time
            
            retrieval_results.append(QuerySolution(
                question=query, 
                docs=top_k_docs, 
                doc_scores=sorted_doc_scores[:num_to_retrieve].tolist()
            ))
        
        # Set batch timing for backward compatibility
        batch_end_time = time.time()
        self.all_retrieval_time = batch_end_time - batch_start_time
        
        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Last Query Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Last Query PPR Time {self.ppr_time:.2f}s")
        
        return retrieval_results

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """QA method để generate answers theo đúng HippoRAG gốc"""
        all_response_messages = []
        all_metadata = []
        
        for query_solution in queries:
            query = query_solution.question
            retrieved_passages = [(doc, score) for doc, score in zip(query_solution.docs, query_solution.doc_scores)]
            
            # Generate answer
            answer = self.generate_answer(query, retrieved_passages)
            query_solution.answer = answer
            
            all_response_messages.append(answer)
            all_metadata.append({"num_docs": len(retrieved_passages)})
        
        return queries, all_response_messages, all_metadata

    def generate_answer(self, query: str, retrieved_passages: List[Tuple[str, float]]) -> str:
        """✅ REFACTORED: Generate answer using PromptTemplateManager"""
        if not retrieved_passages:
            return "Không tìm thấy thông tin liên quan đến câu hỏi của bạn."
        
        # Take top qa_top_k passages for QA
        top_passages = retrieved_passages[:self.global_config.qa_top_k]
        context = "\n".join([text for text, _ in top_passages])
        
        # ✅ Use PromptTemplateManager instead of hardcoded prompts
        prompt_template = self.prompt_manager.render('qa', context=context, query=query)
        
        # Convert messages to prompt format for our API
        prompt_parts = []
        for msg in prompt_template:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "example":
                prompt_parts.append(f"Example: {content}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\Example:"
        
        # Use native Ollama API format
        payload = {
            "model": self.global_config.llm_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.global_config.temperature,
                "num_predict": self.global_config.max_new_tokens,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(self.global_config.llm_base_url, json=payload, timeout=120)
            response.raise_for_status()
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                return answer
            else:
                logger.error(f"LLM API request failed with status code: {response.status_code}")
                return "Đã xảy ra lỗi khi tạo câu trả lời."
                
        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            return "Đã xảy ra lỗi khi tạo câu trả lời."

    def reset_timing(self):
        """Reset timing variables for individual query measurement"""
        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

    def get_last_query_timing(self):
        """Get timing for the last query (not accumulated)"""
        return {
            'ppr_time': self.ppr_time,
            'rerank_time': self.rerank_time,
            'retrieval_time': self.all_retrieval_time
        }

    def rag_qa(self, queries: List[str], gold_docs: List[List[str]] = None, gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        End-to-end question answering với timing reset per query
        """
        # Reset timing trước khi bắt đầu
        self.reset_timing()
        
        query_solutions = self.retrieve(queries)
        query_solutions, all_response_messages, all_metadata = self.qa(query_solutions)
        
        # Attach gold docs and answers if provided
        for i, query_solution in enumerate(query_solutions):
            if gold_docs and i < len(gold_docs):
                query_solution.gold_docs = gold_docs[i]
            if gold_answers and i < len(gold_answers):
                query_solution.gold_answers = gold_answers[i]
        
        return query_solutions, all_response_messages, all_metadata

    def invalidate_retrieval_cache(self):
        """THÊM: Invalidate cached embeddings khi có data mới"""
        self.ready_to_retrieve = False
        self.query_to_embedding = {'triple': {}, 'passage': {}}
        logger.info("Retrieval cache invalidated - embeddings will be recomputed")

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        🔍 GRAPH ANALYTICS: Trích xuất thông tin thống kê chi tiết của knowledge graph theo đúng HippoRAG gốc
        
        Returns:
            Dict chứa các thông tin thống kê về graph:
            - unique_nodes: Số lượng unique nodes
            - unique_edges: Số lượng unique edges  
            - unique_triples: Số lượng unique triples
            - synonym_edges: Số lượng synonym edges
            - node_types: Phân tích theo loại node (chỉ entity và passage)
            - edge_types: Phân tích theo loại edge
        """
        try:
            stats = {}
            
            # 1. Basic Graph Statistics
            stats['unique_nodes'] = self.graph.vcount()
            stats['unique_edges'] = self.graph.ecount()
            
            # 2. Node Type Analysis - THEO HIPPORAG GỐC: CHỈ entity và passage
            node_types = {'entity': 0, 'passage': 0, 'unknown': 0}
            
            if self.graph.vcount() > 0:
                for vertex in self.graph.vs:
                    node_name = vertex['name']
                    if node_name.startswith('entity-'):
                        node_types['entity'] += 1
                    elif node_name.startswith('chunk-'):
                        node_types['passage'] += 1
                    else:
                        node_types['unknown'] += 1
            
            stats['node_types'] = node_types
            
            # 3. Unique Triples Count - THEO Facts chỉ lưu trong embedding store, không trong graph
            stats['unique_triples'] = len(self.fact_embedding_store.get_all_ids()) if hasattr(self, 'fact_embedding_store') else 0
            
            # 4. Edge Type Analysis - THEO HIPPORAG GỐC
            synonym_edges_count = 0
            edge_types = {'entity_entity_edge': 0, 'passage_entity_edge': 0, 'synonym_edge': 0, 'unknown_edge': 0}
            
            if self.graph.ecount() > 0:
                for edge in self.graph.es:
                    source_name = self.graph.vs[edge.source]['name']
                    target_name = self.graph.vs[edge.target]['name']
                    weight = edge.get('weight', 1.0)
                    
                    # Classify edge types theo HippoRAG gốc
                    if (source_name.startswith('entity-') and target_name.startswith('entity-')):
                        if weight != 1.0:  # Synonym edges have similarity weights
                            edge_types['synonym_edge'] += 1
                            synonym_edges_count += 1
                        else:  # Direct entity-entity edges from triples
                            edge_types['entity_entity_edge'] += 1
                    elif (source_name.startswith('chunk-') and target_name.startswith('entity-')) or \
                         (source_name.startswith('entity-') and target_name.startswith('chunk-')):
                        edge_types['passage_entity_edge'] += 1
                    else:
                        edge_types['unknown_edge'] += 1
            
            stats['synonym_edges'] = synonym_edges_count
            stats['edge_types'] = edge_types
            
            # 5. Additional Statistics
            stats['graph_density'] = (2.0 * stats['unique_edges']) / (stats['unique_nodes'] * (stats['unique_nodes'] - 1)) if stats['unique_nodes'] > 1 else 0
            stats['avg_degree'] = (2.0 * stats['unique_edges']) / stats['unique_nodes'] if stats['unique_nodes'] > 0 else 0
            
            # 6. Store Statistics  
            stats['store_statistics'] = {
                'entities_in_store': len(self.entity_embedding_store.get_all_ids()) if hasattr(self, 'entity_embedding_store') else 0,
                'passages_in_store': len(self.chunk_embedding_store.get_all_ids()) if hasattr(self, 'chunk_embedding_store') else 0,
                'facts_in_store': len(self.fact_embedding_store.get_all_ids()) if hasattr(self, 'fact_embedding_store') else 0
            }
            
            # 7. THÊM: Duplicate Statistics
            if hasattr(self, 'duplicate_stats'):
                stats['duplicate_statistics'] = self.get_duplicate_statistics()
            else:
                stats['duplicate_statistics'] = {
                    'status': 'No duplicate tracking performed',
                    'node_duplicates': 0,
                    'edge_duplicates': 0,
                    'total_duplicates': 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {
                'unique_nodes': 0,
                'unique_edges': 0, 
                'unique_triples': 0,
                'synonym_edges': 0,
                'error': str(e)
            }

    def log_graph_statistics(self, log_func=logger.info):
        """
        Log thông tin thống kê chi tiết của knowledge graph
        Input: log_func (callable) - logging function to use
        Output: None
        Chức năng: Log detailed graph statistics
        """
        if not hasattr(self, 'graph') or self.graph is None:
            log_func("Knowledge Graph not initialized")
            return
            
        try:
            graph_stats = self.get_graph_statistics()
            log_func("KNOWLEDGE GRAPH STATISTICS")
            log_func("=" * 50)
            
            # Basic metrics
            log_func("BASIC GRAPH METRICS:")
            log_func(f"  Nodes (N): {graph_stats['unique_nodes']}")
            log_func(f"  Edges (E): {graph_stats['unique_edges']}")
            log_func(f"  Facts (triples): {graph_stats['unique_triples']}")
            
            log_func("GRAPH PROPERTIES:")
            log_func(f"  Density: {graph_stats['graph_density']:.6f}")
            log_func(f"  Average Degree: {graph_stats['avg_degree']:.2f}")
            
            # Node types
            if 'node_types' in graph_stats:
                log_func(f"  Entity Nodes: {graph_stats['node_types']['entity']}")
                log_func(f"  Passage Nodes: {graph_stats['node_types']['passage']}")
                log_func(f"  Unknown Nodes: {graph_stats['node_types']['unknown']}")
            
            # Edge types
            if 'edge_types' in graph_stats:
                log_func("Edge Types:")  
                for edge_type, count in graph_stats['edge_types'].items():
                    percentage = (count / graph_stats['unique_edges'] * 100) if graph_stats['unique_edges'] > 0 else 0
                    # UPDATED: Format edge type names
                    if edge_type == 'entity_entity_edge':
                        display_name = 'Entity-Entity Edges'
                    elif edge_type == 'passage_entity_edge':
                        display_name = 'Passage-Entity Edges'
                    elif edge_type == 'synonym_edge':
                        display_name = 'Synonym Edges (E\')'
                    else:
                        display_name = edge_type.replace('_', ' ').title()
                        
                    log_func(f"  {display_name}: {count} ({percentage:.1f}%)")
            
            log_func("EMBEDDING STORE STATISTICS:")
            if hasattr(self, 'entity_embedding_store'):
                log_func(f"  Entity Store Count: {self.entity_embedding_store.get_entity_count()}")
            if hasattr(self, 'chunk_embedding_store'):  
                log_func(f"  Passage Store Count: {self.chunk_embedding_store.get_passage_count()}")
            if hasattr(self, 'fact_embedding_store'):
                log_func(f"  Fact Store Count: {self.fact_embedding_store.get_fact_count()}")
                
            log_func("=" * 50)
            
        except Exception as e:
            log_func(f"Error logging graph statistics: {e}")

    def print_graph_statistics(self):
        """
        In ra thông tin thống kê chi tiết của knowledge graph (backward compatibility)
        Input: None
        Output: None (prints to stdout)
        Chức năng: Print graph statistics to console
        """
        self.log_graph_statistics(log_func=print)

    def get_detailed_synonymy_analysis(self) -> Dict[str, Any]:
        """
        🔍 DETAILED SYNONYMY ANALYSIS: Phân tích chi tiết về synonym edges
        
        Returns:
            Dict chứa thông tin chi tiết về synonym edges
        """
        try:
            analysis = {
                'synonym_pairs': [],
                'similarity_distribution': [],
                'high_similarity_pairs': [],
                'statistics': {}
            }
            
            if self.graph.ecount() == 0:
                return analysis
            
            synonym_threshold = getattr(self.global_config, 'synonymy_edge_sim_threshold', 0.9)
            
            for edge in self.graph.es:
                source_name = self.graph.vs[edge.source]['name']
                target_name = self.graph.vs[edge.target]['name']
                weight = edge.get('weight', 0)
                
                # Identify synonym edges (entity-entity with similarity weight)
                if (source_name.startswith('entity-') and target_name.startswith('entity-') and 
                    weight != 1.0 and weight > synonym_threshold):
                    
                    # Get entity texts
                    try:
                        source_entity = self.entity_embedding_store.get_row(source_name)['content']
                        target_entity = self.entity_embedding_store.get_row(target_name)['content']
                        
                        synonym_pair = {
                            'entity_1': source_entity,
                            'entity_2': target_entity,
                            'similarity': weight,
                            'hash_1': source_name,
                            'hash_2': target_name
                        }
                        
                        analysis['synonym_pairs'].append(synonym_pair)
                        analysis['similarity_distribution'].append(weight)
                        
                        if weight > synonym_threshold + 0.05:  # High similarity
                            analysis['high_similarity_pairs'].append(synonym_pair)
                            
                    except Exception as e:
                        logger.warning(f"Could not retrieve entity text for synonym analysis: {e}")
            
            # Calculate statistics
            if analysis['similarity_distribution']:
                similarities = analysis['similarity_distribution']
                analysis['statistics'] = {
                    'total_synonym_pairs': len(analysis['synonym_pairs']),
                    'avg_similarity': np.mean(similarities),
                    'max_similarity': np.max(similarities),
                    'min_similarity': np.min(similarities),
                    'std_similarity': np.std(similarities),
                    'high_similarity_count': len(analysis['high_similarity_pairs']),
                    'threshold_used': synonym_threshold
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in detailed synonymy analysis: {e}")
            return {'error': str(e)}

# Compatibility alias for existing code
HippoRAGComplete = HippoRAG

# Export cho backward compatibility
__all__ = ['HippoRAG', 'HippoRAGComplete', 'QuerySolution', 'BaseConfig'] 