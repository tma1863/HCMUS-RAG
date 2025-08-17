"""
Graph Analytics Module

Input: HippoRAG graph objects and configurations
Output: Detailed statistics and analysis results
Chức năng: Analyze graph structure, connectivity, and semantic relationships
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class GraphAnalytics:
    """
    Provides methods to:
    - Extract basic graph statistics
    - Analyze node and edge distributions  
    - Generate detailed synonym analysis
    - Export statistics to various formats
    - Create comparison reports
    """
    
    def __init__(self, hipporag_instance):
        """
        Initialize GraphAnalytics with HippoRAG instance
        
        Args:
            hipporag_instance: Instance of HippoRAG class
        """
        self.hipporag = hipporag_instance
        logger.info("GraphAnalytics initialized")
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics including all metrics
        Input: graph, entity_store, fact_store, chunk_store
        Output: Dict - comprehensive statistics
        """
        try:
            stats = {}
            
            # Basic statistics from HippoRAG
            basic_stats = self.hipporag.get_graph_statistics()
            stats.update(basic_stats)
            
            # Enhanced node analysis
            stats['node_analysis'] = self._analyze_node_connectivity()
            
            # Enhanced edge analysis
            stats['edge_analysis'] = self._analyze_edge_patterns()
            
            # Triple quality analysis
            stats['triple_analysis'] = self._analyze_triple_quality()
            
            # Graph structural metrics
            stats['structural_metrics'] = self._calculate_structural_metrics()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting comprehensive statistics: {e}")
            return {'error': str(e)}
    
    def _analyze_node_connectivity(self) -> Dict[str, Any]:
        """Analyze node connectivity patterns"""
        try:
            analysis = {
                'degree_distribution': [],
                'isolated_nodes': 0,
                'hub_nodes': [],
                'connectivity_stats': {}
            }
            
            if self.hipporag.graph.vcount() == 0:
                return analysis
            
            # Calculate degree for each node
            degrees = self.hipporag.graph.degree()
            analysis['degree_distribution'] = degrees
            
            # Find isolated nodes (degree = 0)
            analysis['isolated_nodes'] = sum(1 for d in degrees if d == 0)
            
            # Find hub nodes (top 10% by degree)
            if degrees:
                degree_threshold = np.percentile(degrees, 90)
                hub_indices = [i for i, d in enumerate(degrees) if d >= degree_threshold and d > 0]
                
                for idx in hub_indices[:10]:  # Top 10 hub nodes
                    node_name = self.hipporag.graph.vs[idx]['name']
                    analysis['hub_nodes'].append({
                        'node_id': node_name,
                        'degree': degrees[idx],
                        'node_type': self._get_node_type(node_name)
                    })
            
            # Connectivity statistics
            if degrees:
                analysis['connectivity_stats'] = {
                    'avg_degree': np.mean(degrees),
                    'max_degree': np.max(degrees),
                    'min_degree': np.min(degrees),
                    'std_degree': np.std(degrees),
                    'median_degree': np.median(degrees)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing node connectivity: {e}")
            return {'error': str(e)}
    
    def _analyze_edge_patterns(self) -> Dict[str, Any]:
        """Analyze edge patterns and weights"""
        try:
            analysis = {
                'weight_distribution': [],
                'edge_type_stats': {},
                'similarity_patterns': {}
            }
            
            if self.hipporag.graph.ecount() == 0:
                return analysis
            
            # Analyze edge weights
            weights = []
            edge_types = {'fact_edges': [], 'passage_edges': [], 'synonym_edges': []}
            
            for edge in self.hipporag.graph.es:
                weight = edge.get('weight', 1.0)
                weights.append(weight)
                
                source_name = self.hipporag.graph.vs[edge.source]['name']
                target_name = self.hipporag.graph.vs[edge.target]['name']
                
                # Classify edges
                if self._is_synonym_edge(source_name, target_name, weight):
                    edge_types['synonym_edges'].append(weight)
                elif self._is_fact_edge(source_name, target_name):
                    edge_types['fact_edges'].append(weight)
                elif self._is_passage_edge(source_name, target_name):
                    edge_types['passage_edges'].append(weight)
            
            analysis['weight_distribution'] = weights
            
            # Edge type statistics
            for edge_type, type_weights in edge_types.items():
                if type_weights:
                    analysis['edge_type_stats'][edge_type] = {
                        'count': len(type_weights),
                        'avg_weight': np.mean(type_weights),
                        'max_weight': np.max(type_weights),
                        'min_weight': np.min(type_weights)
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing edge patterns: {e}")
            return {'error': str(e)}
    
    def _analyze_triple_quality(self) -> Dict[str, Any]:
        """Analyze quality and patterns of extracted triples"""
        try:
            analysis = {
                'triple_length_distribution': [],
                'entity_frequency': {},
                'predicate_frequency': {},
                'quality_metrics': {}
            }
            
            if not hasattr(self.hipporag, 'fact_embedding_store'):
                return analysis
            
            # Get all facts
            fact_ids = self.hipporag.fact_embedding_store.get_all_ids()
            fact_data = self.hipporag.fact_embedding_store.get_rows(fact_ids)
            
            entity_counts = {}
            predicate_counts = {}
            triple_lengths = []
            
            for fact_id, fact_info in fact_data.items():
                try:
                    # Parse fact content
                    fact_content = fact_info['content']
                    triple = eval(fact_content)  # HippoRAG stores as string representation
                    
                    if len(triple) == 3:
                        subject, predicate, obj = triple
                        
                        # Count entities
                        entity_counts[subject] = entity_counts.get(subject, 0) + 1
                        entity_counts[obj] = entity_counts.get(obj, 0) + 1
                        
                        # Count predicates
                        predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
                        
                        # Triple length (character count)
                        triple_length = len(subject) + len(predicate) + len(obj)
                        triple_lengths.append(triple_length)
                        
                except Exception as e:
                    logger.warning(f"Error parsing triple {fact_id}: {e}")
            
            # Most frequent entities and predicates
            analysis['entity_frequency'] = dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20])
            analysis['predicate_frequency'] = dict(sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)[:20])
            analysis['triple_length_distribution'] = triple_lengths
            
            # Quality metrics
            if triple_lengths:
                analysis['quality_metrics'] = {
                    'total_unique_entities': len(entity_counts),
                    'total_unique_predicates': len(predicate_counts),
                    'avg_triple_length': np.mean(triple_lengths),
                    'entity_reuse_rate': sum(entity_counts.values()) / len(entity_counts) if entity_counts else 0,
                    'predicate_diversity': len(predicate_counts) / len(fact_data) if fact_data else 0
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing triple quality: {e}")
            return {'error': str(e)}
    
    def _calculate_structural_metrics(self) -> Dict[str, Any]:
        """Calculate advanced structural metrics"""
        try:
            metrics = {}
            
            if self.hipporag.graph.vcount() == 0:
                return metrics
            
            # Basic structural metrics
            metrics['clustering_coefficient'] = self.hipporag.graph.transitivity_undirected()
            metrics['diameter'] = self.hipporag.graph.diameter() if self.hipporag.graph.is_connected() else float('inf')
            metrics['radius'] = self.hipporag.graph.radius() if self.hipporag.graph.is_connected() else float('inf')
            
            # Component analysis
            components = self.hipporag.graph.components()
            metrics['connected_components'] = len(components)
            metrics['largest_component_size'] = max(len(component) for component in components) if components else 0
            
            # Centrality measures (for smaller graphs)
            if self.hipporag.graph.vcount() < 1000:  # Only for manageable sizes
                try:
                    betweenness = self.hipporag.graph.betweenness()
                    closeness = self.hipporag.graph.closeness()
                    
                    metrics['avg_betweenness'] = np.mean(betweenness)
                    metrics['avg_closeness'] = np.mean(closeness)
                    metrics['max_betweenness'] = np.max(betweenness)
                    metrics['max_closeness'] = np.max(closeness)
                    
                except Exception as e:
                    logger.warning(f"Error calculating centrality metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating structural metrics: {e}")
            return {'error': str(e)}
    
    def _get_node_type(self, node_name: str) -> str:
        """Determine node type from node name"""
        if node_name.startswith('entity-'):
            return 'entity'
        elif node_name.startswith('fact-'):
            return 'fact'
        elif node_name.startswith('chunk-'):
            return 'passage'
        else:
            return 'unknown'
    
    def _is_synonym_edge(self, source_name: str, target_name: str, weight: float) -> bool:
        """Check if edge is a synonym edge"""
        return (source_name.startswith('entity-') and target_name.startswith('entity-') and 
                weight != 1.0)
    
    def _is_fact_edge(self, source_name: str, target_name: str) -> bool:
        """Check if edge is a fact edge"""
        return ((source_name.startswith('entity-') and target_name.startswith('fact-')) or
                (source_name.startswith('fact-') and target_name.startswith('entity-')))
    
    def _is_passage_edge(self, source_name: str, target_name: str) -> bool:
        """Check if edge is a passage edge"""
        return ((source_name.startswith('chunk-') and target_name.startswith('entity-')) or
                (source_name.startswith('entity-') and target_name.startswith('chunk-')))
    
    def create_comparison_report(self, other_stats: Dict[str, Any], output_path: str = None) -> str:
        """
        Create comparison report between two graph statistics
        
        Args:
            other_stats: Statistics from another graph for comparison
            output_path: Path to save comparison report
            
        Returns:
            Path to saved report
        """
        try:
            current_stats = self.get_comprehensive_statistics()
            
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'current_graph': current_stats,
                'comparison_graph': other_stats,
                'differences': self._calculate_differences(current_stats, other_stats)
            }
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"graph_comparison_{timestamp}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Comparison report saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating comparison report: {e}")
            raise
    
    def _calculate_differences(self, stats1: Dict, stats2: Dict) -> Dict[str, Any]:
        """Calculate differences between two statistics dictionaries"""
        differences = {}
        
        # Compare basic metrics
        basic_metrics = ['unique_nodes', 'unique_edges', 'unique_triples', 'synonym_edges']
        for metric in basic_metrics:
            if metric in stats1 and metric in stats2:
                diff = stats1[metric] - stats2[metric]
                pct_change = (diff / stats2[metric] * 100) if stats2[metric] > 0 else 0
                differences[metric] = {
                    'absolute_difference': diff,
                    'percentage_change': pct_change,
                    'current': stats1[metric],
                    'comparison': stats2[metric]
                }
        
        return differences
    
    def export_to_csv(self, output_dir: str = None) -> List[str]:
        """
        Export graph statistics to CSV files
        
        Args:
            output_dir: Directory to save CSV files
            
        Returns:
            List of paths to saved CSV files
        """
        try:
            import os
            
            if output_dir is None:
                output_dir = self.hipporag.working_dir
            
            os.makedirs(output_dir, exist_ok=True)
            
            stats = self.get_comprehensive_statistics()
            saved_files = []
            
            # Basic statistics CSV
            basic_data = {
                'Metric': ['Unique Nodes', 'Unique Edges', 'Unique Triples', 'Synonym Edges', 'Graph Density', 'Average Degree'],
                'Value': [
                    stats.get('unique_nodes', 0),
                    stats.get('unique_edges', 0),
                    stats.get('unique_triples', 0),
                    stats.get('synonym_edges', 0),
                    stats.get('graph_density', 0),
                    stats.get('avg_degree', 0)
                ]
            }
            
            basic_df = pd.DataFrame(basic_data)
            basic_path = os.path.join(output_dir, 'graph_basic_statistics.csv')
            basic_df.to_csv(basic_path, index=False)
            saved_files.append(basic_path)
            
            # Node types CSV
            if 'node_types' in stats:
                node_data = []
                for node_type, count in stats['node_types'].items():
                    percentage = (count / stats['unique_nodes'] * 100) if stats['unique_nodes'] > 0 else 0
                    node_data.append({
                        'Node_Type': node_type,
                        'Count': count,
                        'Percentage': percentage
                    })
                
                node_df = pd.DataFrame(node_data)
                node_path = os.path.join(output_dir, 'graph_node_distribution.csv')
                node_df.to_csv(node_path, index=False)
                saved_files.append(node_path)
            
            # Edge types CSV
            if 'edge_types' in stats:
                edge_data = []
                for edge_type, count in stats['edge_types'].items():
                    percentage = (count / stats['unique_edges'] * 100) if stats['unique_edges'] > 0 else 0
                    edge_data.append({
                        'Edge_Type': edge_type,
                        'Count': count,
                        'Percentage': percentage
                    })
                
                edge_df = pd.DataFrame(edge_data)
                edge_path = os.path.join(output_dir, 'graph_edge_distribution.csv')
                edge_df.to_csv(edge_path, index=False)
                saved_files.append(edge_path)
            
            logger.info(f"Graph statistics exported to {len(saved_files)} CSV files in {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise

    def log_quick_analysis(self, output_func=logger.info):
        """
        Log basic graph analysis using logger
        Input: output_func (callable) - logging function to use
        Output: None
        Chức năng: Log basic graph analysis statistics
        """
        stats = self.get_comprehensive_statistics()
        
        output_func("QUICK GRAPH ANALYSIS")
        output_func("=" * 30)
        output_func(f"Nodes: {stats['unique_nodes']:,}")
        output_func(f"Edges: {stats['unique_edges']:,}")
        output_func(f"Triples: {stats['unique_triples']:,}")
        output_func(f"Density: {stats['graph_density']:.6f}")
        output_func(f"Connected: {stats['is_connected']}")
        
        if 'node_types' in stats:
            output_func(f"Entity Nodes: {stats['node_types'].get('entity', 0)}")
            output_func(f"Passage Nodes: {stats['node_types'].get('passage', 0)}")
        
        output_func("=" * 30)

    def print_quick_analysis(self):
        """
        Print basic graph analysis (backward compatibility)
        Input: None
        Output: None (prints to stdout)
        Chức năng: Print basic graph statistics to console
        """
        self.log_quick_analysis(output_func=print)

def quick_graph_analysis(hipporag_instance, log_level: str = 'info', use_print: bool = False) -> Dict[str, Any]:
    """
    Quick function to get basic graph analysis
    Input: hipporag_instance, log_level, use_print
    Output: Dict - basic graph statistics
    Chức năng: Get and display basic graph analysis
    """
    try:
        stats = hipporag_instance.get_graph_statistics()
        
        # Get appropriate output function
        if use_print:
            output_func = print
        else:
            output_func = getattr(logger, log_level.lower(), logger.info)
        
        # Log/print quick summary
        output_func("QUICK GRAPH ANALYSIS")
        output_func("=" * 40)
        output_func(f"Nodes: {stats['unique_nodes']:,}")
        output_func(f"Edges: {stats['unique_edges']:,}")
        output_func(f"Triples: {stats['unique_triples']:,}")
        output_func(f"Synonyms: {stats['synonym_edges']:,}")
        output_func("=" * 40)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in quick graph analysis: {e}")
        return {'error': str(e)}

def log_graph_analysis(hipporag_instance, log_level: str = 'info') -> Dict[str, Any]:
    """
    Log basic graph analysis using logger
    Input: hipporag_instance, log_level
    Output: Dict - basic graph statistics
    Chức năng: Log basic graph analysis using logger
    """
    return quick_graph_analysis(hipporag_instance, log_level=log_level, use_print=False)

def print_graph_analysis(hipporag_instance) -> Dict[str, Any]:
    """
    Print basic graph analysis (backward compatibility)
    Input: hipporag_instance
    Output: Dict - basic graph statistics
    Chức năng: Print basic graph analysis to console
    """
    return quick_graph_analysis(hipporag_instance, use_print=True) 