ner_system = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""


one_shot_ner_paragraph = """
Elementary Financial Mathematics (ID: MTH10201) is a Compulsory subject conducted during the odd semester.\n
The course is instructed by Phan Thi Phuong, and it It requires prior completion of Calculus 1A, Calculus 2A. \n
Upon completion, students will achieve: Equip students with the basic knowledge of finance and financial mathematics for discrete non-random models.\n
The course includes: Including the theory of interest rates, money chains, forms of borrowing, appraisal of investment projects, valuation of bonds and stocks.
"""


one_shot_ner_output = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

one_shot_ner_output = """{"named_entities":
    ["Elementary Financial Mathematics", "MTH10201", "Compulsory subject", "odd semester", "Phan Thi Phuong", "Calculus 1A", "Calculus 2A", "finance", "financial mathematics for discrete non-random models", "theory of interest rates", "money chains", "borrowing", "appraisal of investment projects", "valuation of bonds and stocks"]
}
"""

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]