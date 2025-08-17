import json
import requests
import logging
import re
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.hipporag_utils import NerRawOutput, TripleRawOutput, compute_mdhash_id, filter_invalid_triples
from hipporag_prompts import PromptTemplateManager
from config.config import config
import os
import time

logger = logging.getLogger(__name__)

# Configuration for parallel processing - giống HippoRAG
MAX_WORKERS = min(32, (os.cpu_count() or 1))

def _extract_ner_from_response(real_response):
    """Extract NER from response - improved version with robust JSON parsing"""
    try:
        # Clean the response first
        clean_response = real_response.strip()
        
        # Remove markdown code blocks if present
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:]
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()
        
        # Try to parse as JSON first
        if clean_response.startswith('{'):
            try:
                # Find the end of the first JSON object to handle extra data
                brace_count = 0
                json_end = -1
                for i, char in enumerate(clean_response):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    json_part = clean_response[:json_end]
                    data = json.loads(json_part)
                    return data.get("named_entities", [])
                else:
                    # Fallback to full string
                    data = json.loads(clean_response)
                    return data.get("named_entities", [])
            except json.JSONDecodeError:
                # Fall through to regex approach
                pass
        
        # Fallback: regex pattern
        pattern = r'"named_entities"\s*:\s*\[(.*?)\]'
        match = re.search(pattern, clean_response, re.DOTALL)
        if match:
            entities_str = match.group(1)
            # Extract quoted strings
            entities = re.findall(r'"([^"]*)"', entities_str)
            return entities
        
        return []
    except Exception as e:
        logger.warning(f"Error extracting NER: {e}")
        return []

def _extract_triples_from_response(real_response):
    """Extract triples từ response - ULTRA ROBUST version with multiple fallbacks"""
    logger.debug(f"Processing response of length: {len(real_response)}")
    
    try:
        # Clean the response first
        clean_response = real_response.strip()
        
        # Remove markdown code blocks
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:]
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()
        
        # STRATEGY 1: Extract complete triples using regex FIRST
        # This is more reliable than trying to parse broken JSON
        triple_pattern = r'\[\s*"([^"]*?)"\s*,\s*"([^"]*?)"\s*,\s*"([^"]*?)"\s*\]'
        matches = re.findall(triple_pattern, clean_response, re.DOTALL)
        
        if matches:
            # Filter out empty or invalid matches
            valid_triples = []
            for match in matches:
                subject, predicate, obj = match
                if subject.strip() and predicate.strip() and obj.strip():
                    valid_triples.append([subject.strip(), predicate.strip(), obj.strip()])
            
            if valid_triples:
                logger.debug(f"Extracted {len(valid_triples)} triples using regex")
                return valid_triples
        
        # STRATEGY 2: Try JSON parsing with auto-completion
        json_start = clean_response.find('{')
        if json_start >= 0:
            # Find potential JSON end
            json_content = clean_response[json_start:]
            
            # Try to find existing closing brace
            brace_count = 0
            json_end = -1
            for i, char in enumerate(json_content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > 0:
                # Complete JSON found
                json_part = json_content[:json_end]
            else:
                # Incomplete JSON - try to complete it intelligently
                json_part = json_content
                
                # Remove incomplete trailing content
                # Look for last complete triple
                last_complete_triple = -1
                bracket_depth = 0
                in_string = False
                escape_next = False
                
                for i, char in enumerate(json_part):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == '[':
                            bracket_depth += 1
                        elif char == ']':
                            bracket_depth -= 1
                            if bracket_depth == 1:  # End of a triple array (inside triples array)
                                last_complete_triple = i + 1
                
                if last_complete_triple > 0:
                    json_part = json_part[:last_complete_triple]
                
                # Add missing closing brackets and braces
                open_brackets = json_part.count('[') - json_part.count(']')
                open_braces = json_part.count('{') - json_part.count('}')
                
                if open_brackets > 0:
                    json_part += ']' * open_brackets
                if open_braces > 0:
                    json_part += '}' * open_braces
            
            try:
                data = json.loads(json_part)
                triples = data.get("triples", [])
                if triples:
                    logger.debug(f"Extracted {len(triples)} triples using JSON parsing")
                    return triples
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed: {e}")
        
        # STRATEGY 3: Parse line by line for triples
        lines = clean_response.split('\n')
        extracted_triples = []
        
        for line in lines:
            line = line.strip()
            # Look for patterns like: ["subject", "predicate", "object"],
            line_match = re.search(r'\[\s*"([^"]*?)"\s*,\s*"([^"]*?)"\s*,\s*"([^"]*?)"\s*\]', line)
            if line_match:
                subject, predicate, obj = line_match.groups()
                if subject.strip() and predicate.strip() and obj.strip():
                    extracted_triples.append([subject.strip(), predicate.strip(), obj.strip()])
        
        if extracted_triples:
            logger.debug(f"Extracted {len(extracted_triples)} triples using line-by-line parsing")
            return extracted_triples
        
        logger.warning("No triples could be extracted using any method")
        return []
        
    except Exception as e:
        logger.error(f"Failed to extract triples: {e}")
        return []

def fix_broken_generated_json(raw_response):
    """Fix broken JSON response with improved error handling"""
    clean_response = raw_response.strip()
    
    # Remove markdown code blocks
    if clean_response.startswith('```json'):
        clean_response = clean_response[7:]
    if clean_response.endswith('```'):
        clean_response = clean_response[:-3]
    clean_response = clean_response.strip()
    
    # Handle truncated responses - find the last complete JSON structure
    try:
        # Try to find the end of the last complete JSON object
        brace_count = 0
        bracket_count = 0
        last_complete_pos = -1
        
        for i, char in enumerate(clean_response):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            
            # If we have balanced braces and brackets, this could be a complete JSON
            if brace_count == 0 and bracket_count == 0 and i > 0:
                last_complete_pos = i + 1
        
        if last_complete_pos > 0:
            clean_response = clean_response[:last_complete_pos]
    except:
        pass  # If analysis fails, continue with original approach
    
    # Try to fix common JSON issues
    if not clean_response.endswith('}') and not clean_response.endswith(']'):
        if '"triples"' in clean_response:
            clean_response += ']}'
        elif '"named_entities"' in clean_response:
            clean_response += ']}'
        else:
            clean_response += '}'
    
    return clean_response

class SimpleOpenIE:
    
    def __init__(self, api_url=None, model_name=None):
        # Sử dụng config từ config.py
        self.api_url = api_url or config.API_URL or config.llm_base_url
        self.model_name = model_name or config.MODEL_NAME or config.llm_name
        # Init prompt template manager như HippoRAG
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        logger.info(f"Initialized SimpleOpenIE with model: {self.model_name}")
        logger.info(f"Using API URL: {self.api_url}")
        logger.info(f"Parallel processing: MAX_WORKERS = {MAX_WORKERS}")
    
    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        """
        Named Entity Recognition following HippoRAG implementation
        """
        # PREPROCESSING - sử dụng prompt template của HippoRAG
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        
        try:
            # LLM INFERENCE
            raw_response, metadata = self._call_llm(ner_input_message)
            
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
                
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))  # Remove duplicates while preserving order
            
        except Exception as e:
            logger.warning(f"NER error for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                unique_entities=[],
                metadata=metadata
            )
        
        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )
    
    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        """
        Triple extraction following HippoRAG implementation
        """
        # PREPROCESSING - sử dụng prompt template của HippoRAG
        named_entity_json = json.dumps({"named_entities": named_entities})
        
        # Match template name with prompts.py
        messages = self.prompt_template_manager.render(
            name='triple_extraction',  # Match with template manager
            passage=passage,
            named_entity_json=named_entity_json
        )
        
        raw_response = ""
        metadata = {}
        
        try:
            # LLM INFERENCE
            raw_response, metadata = self._call_llm(messages)
            
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
                
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)
            
        except Exception as e:
            logger.warning(f"Triple extraction error for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )
        
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )
    
    def _call_llm(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """
        Call Ollama API with proper JSON parsing
        """
        # Convert messages to single prompt for Ollama
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
        
        try:
            # Ollama API format
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 2000,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            # Handle Ollama response properly
            response_text = response.text.strip()
            logger.debug(f"Raw Ollama response: {response_text[:200]}...")
            
            # Parse JSON line by line (Ollama may return multiple JSON objects)
            lines = response_text.split('\n')
            result = None
            
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        parsed_line = json.loads(line)
                        # Take the last complete response
                        if 'response' in parsed_line:
                            result = parsed_line
                    except json.JSONDecodeError:
                        continue
            
            if result is None:
                # Fallback: try to parse entire response
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    # Last resort: try to extract just the first JSON object
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(response_text):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    if json_end > 0:
                        first_json = response_text[:json_end]
                        result = json.loads(first_json)
                    else:
                        raise e
            
            # Extract response content
            raw_response = result.get('response', '').strip()
            metadata = {
                'finish_reason': 'stop' if result.get('done', False) else 'length',
                'prompt_tokens': result.get('prompt_eval_count', 0),
                'completion_tokens': result.get('eval_count', 0),
                'cache_hit': False
            }
            
            logger.debug(f"Extracted response: {raw_response[:100]}...")
            return raw_response, metadata
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            logger.error(f"Response text: {response.text[:500] if 'response' in locals() else 'No response'}")
            raise
    
    def openie(self, chunk_key: str, passage: str) -> Dict[str, any]:
        """
        Complete OpenIE process for HippoRAG: NER + Triple Extraction
        """
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(
            chunk_key=chunk_key, 
            passage=passage, 
            named_entities=ner_output.unique_entities
        )
        return {"ner": ner_output, "triplets": triple_output}
    
    def batch_openie(self, chunks: Dict[str, Dict]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Batch OpenIE for interface with enhanced parallel processing and stats
        """
        start_time = time.time()
        
        logger.info(f"BATCH OPENIE STARTED")
        logger.info(f"Processing {len(chunks)} documents using {MAX_WORKERS} parallel workers")
        logger.info(f"System CPU count: {os.cpu_count()}, Using: {MAX_WORKERS} workers")
        
        ner_results_dict = {}
        triple_results_dict = {}
        
        # Extract passages from chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        
        # Analyze document lengths
        doc_lengths = [len(passage) for passage in chunk_passages.values()]
        logger.info(f"Document stats: min={min(doc_lengths)}, max={max(doc_lengths)}, avg={sum(doc_lengths)/len(doc_lengths):.1f} chars")
        
        # Step 1: Parallel NER
        logger.info("STEP 1: Performing parallel NER...")
        ner_start_time = time.time()
        ner_results_list = []
        total_ner_prompt_tokens = 0
        total_ner_completion_tokens = 0
        successful_ner = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }
            
            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                ner_results_dict[result.chunk_id] = result
                
                metadata = result.metadata
                total_ner_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_ner_completion_tokens += metadata.get('completion_tokens', 0)
                
                if len(result.unique_entities) > 0:
                    successful_ner += 1
                
                pbar.set_postfix({
                    'prompt_tokens': total_ner_prompt_tokens,
                    'completion_tokens': total_ner_completion_tokens,
                    'success_rate': f'{successful_ner}/{len(ner_results_list)}'
                })
        
        ner_duration = time.time() - ner_start_time
        total_entities = sum(len(result.unique_entities) for result in ner_results_list)
        
        logger.info(f"NER COMPLETED in {ner_duration:.1f}s")
        logger.info(f"NER Success: {successful_ner}/{len(chunks)} docs ({successful_ner/len(chunks)*100:.1f}%)")
        logger.info(f"Total entities: {total_entities} (avg: {total_entities/len(chunks):.1f} per doc)")
        logger.info(f"NER Tokens: {total_ner_prompt_tokens} prompt + {total_ner_completion_tokens} completion")
        
        # Step 2: Parallel triple extraction
        logger.info("STEP 2: Performing parallel triple extraction...")
        triple_start_time = time.time()
        total_triple_prompt_tokens = 0
        total_triple_completion_tokens = 0
        successful_triples = 0
        json_errors = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            triple_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                              chunk_passages[ner_result.chunk_id],
                              ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            
            pbar = tqdm(as_completed(triple_futures), total=len(triple_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_dict[result.chunk_id] = result
                
                metadata = result.metadata
                total_triple_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_triple_completion_tokens += metadata.get('completion_tokens', 0)
                
                if len(result.triples) > 0:
                    successful_triples += 1
                
                # Count JSON parsing errors
                if 'JSON decode error' in result.response:
                    json_errors += 1
                
                pbar.set_postfix({
                    'prompt_tokens': total_triple_prompt_tokens,
                    'completion_tokens': total_triple_completion_tokens,
                    'success_rate': f'{successful_triples}/{len(triple_results_dict)}',
                    'json_errors': json_errors
                })
        
        triple_duration = time.time() - triple_start_time
        total_triples = sum(len(result.triples) for result in triple_results_dict.values())
        total_duration = time.time() - start_time
        
        logger.info(f"TRIPLE EXTRACTION COMPLETED in {triple_duration:.1f}s")
        logger.info(f"Triple Success: {successful_triples}/{len(chunks)} docs ({successful_triples/len(chunks)*100:.1f}%)")
        logger.info(f"Total triples: {total_triples} (avg: {total_triples/len(chunks):.1f} per doc)")
        logger.info(f"JSON errors: {json_errors}/{len(chunks)} ({json_errors/len(chunks)*100:.1f}%)")
        logger.info(f"Triple Tokens: {total_triple_prompt_tokens} prompt + {total_triple_completion_tokens} completion")
        
        # Overall stats
        total_tokens = total_ner_prompt_tokens + total_ner_completion_tokens + total_triple_prompt_tokens + total_triple_completion_tokens
        parallel_speedup = (ner_duration + triple_duration) / total_duration if total_duration > 0 else 1
        
        logger.info(f"BATCH OPENIE COMPLETED in {total_duration:.1f}s")
        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Processing speed: {len(chunks)/total_duration:.1f} docs/second")
        logger.info(f"Parallel efficiency: {parallel_speedup:.2f}x vs sequential (estimated)")
        
        return ner_results_dict, triple_results_dict

    def extract_triples_from_batch(self, chunks: List[str]) -> List[List[List[str]]]:
        """
        Extract triples from batch of chunks với parallel processing
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of triples for each chunk, where each triple is [subject, predicate, object]
        """
        logger.info(f"Extracting triples from {len(chunks)} chunks using {MAX_WORKERS} workers")
        
        all_triples = [None] * len(chunks)  # Pre-allocate to maintain order
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create futures for each chunk with index to maintain order
            futures = {
                executor.submit(self._extract_triples_for_single_chunk, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results with progress bar
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Extracting triples")
            for future in pbar:
                chunk_index = futures[future]
                try:
                    chunk_triples = future.result()
                    all_triples[chunk_index] = chunk_triples
                except Exception as e:
                    logger.warning(f"Error extracting triples from chunk {chunk_index}: {e}")
                    all_triples[chunk_index] = []  # Empty triples for failed chunks
        
        logger.info(f"Extracted {sum(len(triples) for triples in all_triples)} total triples")
        return all_triples
    
    def _extract_triples_for_single_chunk(self, chunk_index: int, chunk: str) -> List[List[str]]:
        """
        Helper method để extract triples từ single chunk (for parallel processing)
        
        Args:
            chunk_index: Index of the chunk
            chunk: Text content of the chunk
            
        Returns:
            List of triples for the chunk
        """
        chunk_key = f"chunk_{chunk_index}"
        try:
            # Perform OpenIE
            ner_result = self.ner(chunk_key, chunk)
            triple_result = self.triple_extraction(chunk_key, chunk, ner_result.unique_entities)
            
            # Extract triples
            chunk_triples = triple_result.triples if triple_result.triples else []
            return chunk_triples
            
        except Exception as e:
            logger.warning(f"Error extracting triples from chunk {chunk_index}: {e}")
            return []  # Empty triples for failed chunks
