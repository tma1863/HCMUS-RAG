# hipporag_utils.py
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging
from hashlib import md5

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]

@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]] if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }

def text_processing(text):
    """
    Text processing function
    Input: text (str or List[str])
    Output: processed text (str or List[str])
    Chức năng: Clean và normalize text
    """
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def filter_invalid_triples(triples: List[List[str]]) -> List[List[str]]:
    """
    Filters out invalid and duplicate triples from a list of triples
    Input: triples (List[List[str]]) - list of triples to filter
    Output: List[List[str]] - unique valid triples
    Chức năng: Filter và deduplicate triples, keep only valid 3-element triples
    """
    unique_triples = set()
    valid_triples = []

    for triple in triples:
        if len(triple) != 3: continue  # Skip triples that do not have exactly 3 elements

        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)

    return valid_triples

def reformat_openie_results(corpus_openie_results) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
    """
    Reformat OpenIE results into structured format
    Input: corpus_openie_results - raw OpenIE results
    Output: (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]) - formatted NER and triple results
    Chức năng: Convert raw OpenIE results into structured NER and triple dictionaries
    """
    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute MD5 hash of content string with optional prefix
    Input: content (str), prefix (str, optional)
    Output: str - prefixed MD5 hash
    Chức năng: Generate unique ID using MD5 hash
    """
    return prefix + md5(content.encode()).hexdigest()

def min_max_normalize(x):
    """
    Min-max normalization function
    Input: x (array-like) - values to normalize
    Output: normalized array
    Chức năng: Normalize values to [0,1] range
    """
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def extract_entity_nodes(chunk_triples: List[List[List[str]]]) -> Tuple[List[str], List[List[str]]]:
    """
    Extract entity nodes from chunk triples
    Input: chunk_triples (List[List[List[str]]]) - triples from each chunk
    Output: (List[str], List[List[str]]) - unique graph nodes, chunk entities
    Chức năng: Extract unique entities from triples for graph construction
    """
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[List[List[str]]]) -> List[Tuple[str, str, str]]:
    """
    Flatten chunk triples into unique fact tuples
    Input: chunk_triples (List[List[List[str]]]) - triples from all chunks
    Output: List[Tuple[str, str, str]] - unique fact tuples
    Chức năng: Convert nested triples into flat list of unique tuples
    """
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def all_values_of_same_length(data: dict) -> bool:
    """
    Check if all dictionary values have same length
    Input: data (dict) - dictionary to check
    Output: bool - True if all values have same length
    Chức năng: Validate that all sequences in dict have consistent length
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)
