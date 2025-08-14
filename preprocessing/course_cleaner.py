import re
from typing import List

class CourseCleaner:
    TITLES_TO_REMOVE = ["Assoc. Prof. ", "Dr. ", "MSc. ", "Prof. ", "M.S. ", "M.Sc. ", "Ms. "]
    SPECIAL_CASES = [" Sciences", " University of Science"]
    REPLACEMENT_TEXT = "Assigned lecturers at University of Science"
    PREREQUISITE_PATTERN = r"Recommended prerequisites?:.*|Recommendation:.*|Course requirements:.*|Prerequisite courses:*"

    def __init__(self, filtered_tables: List[List[List[str]]]):
        self.filtered_tables = filtered_tables

    def clean_course_ids(self):
        for table in self.filtered_tables:
            table[0][1] = table[0][1].strip()[-8:]

    def normalize_semesters(self):
        for table in self.filtered_tables:
            digits = [int(ch) for ch in table[2][1] if ch.isdigit()]
            if digits:
                new_value = "even" if any(d % 2 == 0 for d in digits) else "odd"
                table[2][1] = new_value

    def normalize_course_type(self):
        for table in self.filtered_tables:
            table[4][1] = table[4][1].lower().strip()

    def clean_teacher_names(self):
        def clean(name):
            name = name.strip()
            for title in self.TITLES_TO_REMOVE:
                if name.startswith(title):
                    name = name.replace(title, '')
            if any(s in name for s in self.SPECIAL_CASES):
                return self.REPLACEMENT_TEXT
            return name

        for table in self.filtered_tables:
            raw = table[3][1].replace('\n', ',')
            names = [clean(n) for n in raw.split(',') if n.strip()]
            unique = []
            seen = set()
            for name in names:
                if name not in seen:
                    seen.add(name)
                    unique.append(name)
            table[3][1] = ', '.join(unique)


    def clean_course_titles_and_prerequisites(self):
        for table in self.filtered_tables:
            # --- Clean table[1][1] ---
            value_1 = table[1][1]
            if '(' in value_1:
                value_1 = value_1.split('(')[0].strip()
                table[1][1] = value_1

            # --- Clean table[5][1] ---
            value_5 = table[5][1]
            if ': ' in value_5:
                value_5 = value_5.split(': ', 1)[1].strip()
            # Apply regex cleanup
            value_5 = re.sub(self.PREREQUISITE_PATTERN, "", value_5).strip()
            table[5][1] = value_5