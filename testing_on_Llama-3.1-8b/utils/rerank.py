import json
import difflib
import re
import ast
import logging
import requests
from typing import List, Dict, Tuple, Any, Optional, Union
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from pydantic import BaseModel, Field
from ..hipporag_prompts import (
    DEFAULT_DSPy_PROMPT,
    get_rerank_prompt_template,
    get_rerank_templates,
    original_rerank_input_template as rerank_input_template,
    original_rerank_output_template as rerank_output_template
)

logger = logging.getLogger(__name__)

# Lớp định nghĩa cấu trúc Fact chuẩn
class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")

class DSPyFilter:
    """
    DSPyFilter: Lớp thực hiện việc lọc và xếp hạng lại các facts dựa trên sự liên quan đến câu truy vấn.
    Sử dụng original prompts từ HippoRAG gốc.
    """
    def __init__(self, api_url="http://localhost:11434/api/generate", model_name="llama3:8b", prompt=None):
        """
        Khởi tạo DSPyFilter với các thành phần cần thiết.
        
        Args:
            api_url: URL của API LLM để gửi yêu cầu.
            model_name: Tên mô hình LLM được sử dụng.
            prompt: Prompt tùy chỉnh cho DSPyFilter (tùy chọn).
        """
        self.api_url = api_url
        self.model_name = model_name
        self.default_gen_kwargs = {
            'temperature': 0.0,
            'max_tokens': 1024
        }
        
        # Lấy templates từ hipporag_prompts (original HippoRAG)
        self.one_input_template, self.one_output_template = get_rerank_templates()
        
        # Tạo message template từ prompt hoặc sử dụng default từ HippoRAG gốc
        if prompt:
            self.prompt_data = prompt
        else:
            self.prompt_data = get_rerank_prompt_template()
        
        self.message_template = self.make_template()
        
        logger.info(f"Initialized DSPyFilter with model: {model_name} (using original HippoRAG prompts)")
        
    def make_template(self):
        """
        Tạo template message cho DSPyFilter từ prompt đã định nghĩa (original HippoRAG).
        
        Returns:
            List message template cho API LLM.
        """
        # Sử dụng original HippoRAG prompt template
        if isinstance(self.prompt_data, list):
            return deepcopy(self.prompt_data)
        
        # Legacy support cho dict format
        system_prompt = self.prompt_data.get('system', DEFAULT_DSPy_PROMPT[0]['content'])
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Thêm các ví dụ demo
        demos = self.prompt_data.get('demos', [])
        for demo in demos:
            message_template.append({
                "role": "user", 
                "content": self.one_input_template.format(
                    question=demo["question"], 
                    fact_before_filter=demo["fact_before_filter"]
                )
            })
            message_template.append({
                "role": "assistant", 
                "content": self.one_output_template.format(
                    fact_after_filter=demo["fact_after_filter"]
                )
            })
            
        return message_template
    
    def parse_filter(self, response: str) -> List[List[str]]:
        """
        Phân tích phản hồi từ LLM để trích xuất các facts đã lọc.
        Sử dụng nhiều chiến lược parsing để đảm bảo độ ổn định.
        
        Args:
            response: Phản hồi từ LLM cần parse
            
        Returns:
            List[List[str]]: Danh sách các facts, mỗi fact là một list gồm 3 phần tử [subject, predicate, object]
        """
        # Debug logging for response parsing
        logger.info(f"PARSE_FILTER DEBUG - Input response length: {len(response)}")
        logger.info(f"Raw response:\n{response}")
        
        def validate_fact_format(facts: List[Any]) -> List[List[str]]:
            """Kiểm tra và chuẩn hóa format của facts."""
            validated_facts = []
            for fact in facts:
                # Chuyển fact về dạng list nếu là tuple
                if isinstance(fact, tuple):
                    fact = list(fact)
                    
                # Kiểm tra fact có đúng 3 phần tử
                if isinstance(fact, list) and len(fact) == 3:
                    # Chuyển tất cả thành phần về string
                    validated_fact = [str(item).strip() for item in fact]
                    if all(validated_fact):  # Kiểm tra không có phần tử rỗng
                        validated_facts.append(validated_fact)
            return validated_facts

        def try_json_parsing(text: str) -> List[List[str]]:
            """Thử parse JSON với nhiều pattern khác nhau."""
            # Pattern cơ bản: tìm JSON object hoặc array
            json_patterns = [
                r'\{[^{}]*\}',  # JSON object
                r'\[[^\[\]]*\]',  # JSON array
                r'\{.*?\}',  # Greedy JSON object
                r'\[.*?\]'   # Greedy JSON array
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group()
                        data = json.loads(json_str)
                        
                        # Xử lý các trường hợp format khác nhau
                        if isinstance(data, dict) and 'fact' in data:
                            return validate_fact_format(data['fact'])
                        elif isinstance(data, list):
                            return validate_fact_format(data)
                    except json.JSONDecodeError:
                        continue
            return []

        def try_ast_parsing(text: str) -> List[List[str]]:
            """Thử parse sử dụng ast.literal_eval."""
            try:
                # Tìm và làm sạch các list/tuple string
                list_pattern = r'[\[\(].*?[\]\)]'
                matches = re.finditer(list_pattern, text, re.DOTALL)
                
                for match in matches:
                    try:
                        data = ast.literal_eval(match.group())
                        if isinstance(data, (list, tuple)):
                            return validate_fact_format(data)
                    except (ValueError, SyntaxError):
                        continue
            except Exception as e:
                logger.warning(f"AST parsing failed: {e}")
            return []

        def extract_facts_from_sections(text: str) -> List[List[str]]:
            """Trích xuất facts từ các section được đánh dấu."""
            sections = {}
            current_section = None
            buffer = []

            # Tìm các section được đánh dấu
            section_markers = [
                (r'\[\[ ## fact_after_filter ## \]\]', 'fact_after_filter'),
                (r'\[\[ ## facts ## \]\]', 'facts'),
                (r'Facts:', 'facts')
            ]

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue

                matched = False
                for marker_pattern, section_name in section_markers:
                    if re.match(marker_pattern, line, re.IGNORECASE):
                        if current_section and buffer:
                            sections[current_section] = '\n'.join(buffer)
                        current_section = section_name
                        buffer = []
                        matched = True
                        break
                if not matched and current_section:
                    buffer.append(line)
            if current_section and buffer:
                sections[current_section] = '\n'.join(buffer)

            # Ưu tiên section 'fact_after_filter'
            for key in ['fact_after_filter', 'facts']:
                if key in sections:
                    section = sections[key]
                    logger.info(f"Processing {key} section: {section}")
                    json_facts = try_json_parsing(section)
                    if json_facts:
                        return json_facts
                    ast_facts = try_ast_parsing(section)
                    if ast_facts:
                        return ast_facts

            return []

        try:
            # Handle case where response doesn't start with "Facts:"
            response_lines = response.strip().split('\n')
            
            # Find the start of the facts section
            facts_start_idx = 0
            for i, line in enumerate(response_lines):
                if 'facts:' in line.lower() or line.strip().startswith('-') or line.strip().startswith('1.'):
                    facts_start_idx = i
                    break
            
            # Extract facts from the appropriate section
            facts_section = '\n'.join(response_lines[facts_start_idx:])
            facts = self._extract_facts_from_text(facts_section)
            
            if facts:
                logger.info(f"Successfully parsed {len(facts)} facts from response")
                return facts
            else:
                logger.warning("No facts found in response, trying fallback parsing")
                return self._fallback_parse_facts(response)
                
        except Exception as e:
            logger.error(f"Error parsing DSPy response: {e}")
            logger.error(f"Problematic response: {response}")
            return []
    
    def llm_call(self, query: str, facts: Union[List[Tuple], str], max_facts: int = 4) -> str:
        """
        Gọi LLM để xếp hạng lại các facts (sử dụng original HippoRAG prompts).
        """
        # Chuyển đổi facts thành định dạng JSON nếu là danh sách
        if isinstance(facts, list):
            facts_json = json.dumps({"fact": [list(fact) for fact in facts]})
        else:
            facts_json = facts
        
        # Tạo messages cho API
        messages = deepcopy(self.message_template)
        messages.append({
            "role": "user", 
            "content": self.one_input_template.format(
                question=query, 
                fact_before_filter=facts_json
            )
        })
        
        logger.info(f"Sending query to LLM API using original HippoRAG prompts")
        
        # Gọi API ChatCompletion
        payload = {
            "model": self.model_name,
            "messages": messages,
            **self.default_gen_kwargs
        }
        
        try:
            # Kiểm tra nếu đây là Ollama API
            if self.api_url.endswith("/v1/completions"):
                # Sử dụng prompt với Ollama API
                prompt = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
                    elif role == "user":
                        if prompt:
                            prompt += f"{content} [/INST]"
                        else:
                            prompt += f"<s>[INST] {content} [/INST]"
                    elif role == "assistant":
                        prompt += f" {content} </s><s>[INST] "
                
                completion_payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    **self.default_gen_kwargs
                }
                
                response = requests.post(self.api_url, json=completion_payload)
                response.raise_for_status()
                
                if response.status_code == 200:
                    result_text = response.json().get("choices", [{}])[0].get("text", "{}")
                    logger.info(f"Received response of length {len(result_text)}")
                    return result_text
            else:
                # Gọi API OpenAI-compatible với messages
                chat_endpoint = self.api_url.replace("/v1/completions", "/v1/chat/completions")
                response = requests.post(chat_endpoint, json=payload)
                response.raise_for_status()
                
                if response.status_code == 200:
                    result_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    logger.info(f"Received response of length {len(result_text)}")
                    return result_text
                
            logger.error(f"LLM API request failed with status code: {response.status_code}")
            logger.error(f"Response text: {response.text}")
            return "{}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to LLM API failed: {e}")
            return "{}"
        except Exception as e:
            logger.error(f"Unexpected error in llm_call: {e}")
            return "{}"
    
    def rerank(self,
              query: str,
              candidate_items: List[Tuple],
              candidate_indices: List[int],
              len_after_rerank: Optional[int] = None) -> Tuple[List[int], List[Tuple], Dict]:
        """
        Xếp hạng lại các facts dựa trên sự liên quan đến truy vấn (original HippoRAG logic).
        """
        if not candidate_items:
            logger.warning("No candidate items to rerank")
            return [], [], {'confidence': None}
            
        max_facts = min(4, len(candidate_items))
        if len_after_rerank is None:
            len_after_rerank = max_facts
            
        try:
            # Gọi LLM để xếp hạng lại
            response = self.llm_call(query, candidate_items, max_facts)
            generated_facts = self.parse_filter(response)
            
            # Nếu không tìm thấy facts nào, sử dụng các facts đầu tiên
            if not generated_facts and candidate_items:
                logger.warning("No facts returned from LLM, using top candidates instead")
                return (
                    candidate_indices[:len_after_rerank], 
                    candidate_items[:len_after_rerank], 
                    {'confidence': None, 'fallback': True}
                )
            
            # Tìm các facts khớp nhất từ danh sách ứng viên (như HippoRAG gốc)
            result_indices = []
            for generated_fact in generated_facts:
                closest_matches = difflib.get_close_matches(
                    str(generated_fact), 
                    [str(list(item)) for item in candidate_items], 
                    n=1, 
                    cutoff=0.0  # Use 0.0 cutoff như HippoRAG gốc
                )
                
                if closest_matches:
                    closest_match = closest_matches[0]
                    try:
                        match_index = [str(list(item)) for item in candidate_items].index(closest_match)
                        result_indices.append(match_index)
                    except ValueError as e:
                        logger.error(f"Unable to find match: {e}")
            
            # Loại bỏ các chỉ số trùng lặp nhưng giữ thứ tự
            unique_indices = []
            for idx in result_indices:
                if idx not in unique_indices:
                    unique_indices.append(idx)
            result_indices = unique_indices
                
            # Lấy các chỉ số và facts đã xếp hạng lại
            sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
            sorted_candidate_items = [candidate_items[i] for i in result_indices]
            
            # Giới hạn số lượng kết quả
            return (
                sorted_candidate_indices[:len_after_rerank], 
                sorted_candidate_items[:len_after_rerank], 
                {'confidence': None, 'fallback': False}
            )
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            if candidate_items:
                return (
                    candidate_indices[:len_after_rerank], 
                    candidate_items[:len_after_rerank], 
                    {'confidence': None, 'error': str(e)}
                )
            else:
                return [], [], {'confidence': None, 'error': str(e)}
    
    def __call__(self, *args, **kwargs):
        """Cho phép gọi đối tượng như một hàm."""
        return self.rerank(*args, **kwargs) 