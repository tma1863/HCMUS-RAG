"""
HippoRAG prompt templates - Copy exact từ HippoRAG gốc
"""

# =============================================================================
# ORIGINAL HIPPORAG NER PROMPTS
# =============================================================================

# NER Prompts
ner_system = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

one_shot_ner_output = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

ner_prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]

# =============================================================================
# ORIGINAL HIPPORAG TRIPLE EXTRACTION PROMPTS
# =============================================================================

# Triple Extraction Prompts
ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph: {passage}

{named_entity_json}
"""

ner_conditioned_re_input = ner_conditioned_re_frame.format(
    passage=one_shot_ner_paragraph, 
    named_entity_json=one_shot_ner_output
)

ner_conditioned_re_output = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

triple_extraction_prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": ner_conditioned_re_frame}
]

# =============================================================================
# ORIGINAL HIPPORAG RERANK PROMPTS (updated with DSPy Filter prompts)
# =============================================================================

# DSPy Filter system prompt from hipporag_complete.py
dspy_filter_system = "You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {\"fact\": [[\"s1\", \"p1\", \"o1\"], [\"s2\", \"p2\", \"o2\"]]}, and if no facts are relevant, return an empty list, {\"fact\": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information."

# DSPy Filter input template
dspy_filter_input_template = """[[ ## question ## ]]\n${question}\n\n[[ ## fact_before_filter ## ]]\n${fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""

# DSPy Filter output template
dspy_filter_output_template = """[[ ## fact_after_filter ## ]]\n${fact_after_filter}\n\n[[ ## completed ## ]]"""

# Legacy rerank system for backward compatibility
rerank_system = '''Based on the query, please rank the given list of facts by their relevance to the query. 
The most relevant facts should be at the top of the list.
Return the reranked list in the same format.
'''

def get_dspy_filter_prompt_template():
    return [
        {"role": "system", "content": dspy_filter_system},
        {"role": "user", "content": dspy_filter_input_template}
    ]

def get_rerank_prompt_template():
    return [
        {"role": "system", "content": rerank_system},
        {"role": "user", "content": "Query: ${query}\n\nFacts:\n${facts}"}
    ]

def get_rerank_templates():
    return {
        'rerank': get_rerank_prompt_template(),
        'dspy_filter': get_dspy_filter_prompt_template()
    }

# =============================================================================
# ORIGINAL HIPPORAG QA PROMPTS (updated with actual prompts from hipporag_complete.py)
# =============================================================================

# Updated QA system prompt from hipporag_complete.py
rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Although you should internally consider the reasoning process, do not include it in your response. '
    'Only return the final answer starting after "Answer: ", and exclude any "Thought: " or explanation. The final answer do not include keyword "Answer: "'
    'Your answer should be concise, definitive, and match the writing style of the source text. '
)

# One-shot example for QA
one_shot_rag_qa_output = (
    "What are the skills learning outcomes for the Data Structures and Algorithms course? "
    "analyzing algorithms, generalize data, and algorithm settings."
)

def get_qa_prompt_template():
    return [
        {"role": "system", "content": rag_qa_system},
        {"role": "example", "content": one_shot_rag_qa_output},
        {"role": "user", "content": "Context:\n${context}\n\nQuestion: ${query}"}
    ]

# Legacy QA system for backward compatibility
qa_system = '''Based on the retrieved facts, please provide a comprehensive answer to the query.'''

# =============================================================================
# TEMPLATE MANAGER - FIXED to match my_docker_project
# =============================================================================

class PromptTemplateManager:
    """Template manager giống my_docker_project"""
    
    def __init__(self, role_mapping=None):
        self.role_mapping = role_mapping or {"system": "system", "user": "user", "assistant": "assistant"}
    
    def render(self, name: str, **kwargs):
        """Render prompt template with parameters"""
        if name == 'ner':
            # Replace ${passage} with actual passage
            messages = []
            for msg in ner_prompt_template:
                content = msg["content"]
                if "${passage}" in content:
                    content = content.replace("${passage}", kwargs.get("passage", ""))
                messages.append({"role": msg["role"], "content": content})
            return messages
        
        elif name == 'triple_extraction':
            # Replace placeholders in triple extraction template
            passage = kwargs.get("passage", "")
            named_entity_json = kwargs.get("named_entity_json", "")
            
            messages = []
            for msg in triple_extraction_prompt_template:
                content = msg["content"]
                if "{passage}" in content and "{named_entity_json}" in content:
                    content = content.format(passage=passage, named_entity_json=named_entity_json)
                messages.append({"role": msg["role"], "content": content})
            return messages
        
        elif name == 'rerank':
            # Replace placeholders in rerank template
            messages = []
            for msg in get_rerank_prompt_template():
                content = msg["content"]
                if "${query}" in content:
                    content = content.replace("${query}", kwargs.get("query", ""))
                if "${facts}" in content:
                    content = content.replace("${facts}", kwargs.get("facts", ""))
                messages.append({"role": msg["role"], "content": content})
            return messages
            
        elif name == 'dspy_filter':
            # Replace placeholders in DSPy filter template
            messages = []
            for msg in get_dspy_filter_prompt_template():
                content = msg["content"]
                if "${question}" in content:
                    content = content.replace("${question}", kwargs.get("question", ""))
                if "${fact_before_filter}" in content:
                    content = content.replace("${fact_before_filter}", kwargs.get("fact_before_filter", ""))
                messages.append({"role": msg["role"], "content": content})
            return messages
            
        elif name == 'qa':
            # Replace placeholders in QA template
            messages = []
            for msg in get_qa_prompt_template():
                content = msg["content"]
                if "${context}" in content:
                    content = content.replace("${context}", kwargs.get("context", ""))
                if "${query}" in content:
                    content = content.replace("${query}", kwargs.get("query", ""))
                messages.append({"role": msg["role"], "content": content})
            return messages
            
        else:
            raise ValueError(f"Unknown prompt template: {name}")
    
    def get_available_templates(self):
        return ['ner', 'triple_extraction', 'rerank', 'dspy_filter', 'qa']