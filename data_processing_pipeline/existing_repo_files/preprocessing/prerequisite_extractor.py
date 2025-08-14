import pandas as pd
import re
import Levenshtein
from typing import List

class PrerequisiteExtractor:
    def __init__(self, filtered_tables: List[List[List[str]]]):
        self.filtered_tables = filtered_tables
        self.mapping_course_name_origin_list = []
        self.mapping_course_name_lower_list = []
        self.result_dict = {}

    def extract_dataframe(self):
        data = []
        for table in self.filtered_tables:
            course_id = table[0][1]
            values = [row[1] for row in table[1:8]]
            self.result_dict[course_id] = values

            prereqs = re.sub(r'\([^)]*\)', '', values[4].lower()).strip()
            data.append({
                "course_id": course_id,
                "course_name": values[0],
                "prerequisites": prereqs,
                "answer": None
            })
        return pd.DataFrame(data)

    def split_prerequisites(self, df):
        new_rows = []
        for _, row in df.iterrows():
            parts = [p.strip() for p in row['prerequisites'].replace('\n', ',').replace(';', ',').split(',') if p.strip()]
            for p in parts:
                new_row = row.copy()
                new_row['prerequisites'] = p
                new_rows.append(new_row)
        return pd.DataFrame(new_rows)
    
    def standardize_roman_numerals(self, df):
        def replace_arabic_with_roman(text):
            text = re.sub(r'\bgeneral biology 1\b', 'general biology I', text, flags=re.IGNORECASE)
            text = re.sub(r'\bgeneral biology 2\b', 'general biology II', text, flags=re.IGNORECASE)
            return text
        
        df['prerequisites'] = df['prerequisites'].apply(replace_arabic_with_roman)
        return df

    def create_mappings(self):
        mapping = {}
        for table in self.filtered_tables:
            course_id = table[0][1].strip()
            course_name = table[1][1].strip()
            mapping[course_id] = {
                "origin": course_name,
                "lower": course_name.lower()
            }
        self.mapping_course_name_origin_list = [v["origin"] for v in mapping.values()]
        self.mapping_course_name_lower_list = [v["lower"] for v in mapping.values()]

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i  # xóa toàn bộ ký tự
        for j in range(n + 1):
            dp[0][j] = j  # chèn toàn bộ ký tự

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # xóa
                    dp[i][j - 1] + 1,      # chèn
                    dp[i - 1][j - 1] + cost  # thay thế
                )

        return dp[m][n]

    def match_fuzzy(self, df):
        def match(row):
            prereq = row["prerequisites"].lower().strip()

            if prereq == "none":
                row["answer"] = "none"
                return row

            distances = [self.levenshtein_distance(prereq, cand) for cand in self.mapping_course_name_lower_list]
            if not distances:
                return row

            min_idx = min(range(len(distances)), key=lambda i: distances[i])
            row["answer"] = self.mapping_course_name_origin_list[min_idx]
            return row

        df = df.apply(match, axis=1)
        return df.drop_duplicates().reset_index(drop=True)


    @staticmethod
    def adjust_answer_column(df: pd.DataFrame):
        df['prerequisites'] = df['prerequisites'].astype(str)
        df.loc[df['prerequisites'].str.lower().str.contains('credits'), 'answer'] = df['prerequisites']
        
        df.loc[df['course_id'] == 'MTH10616', 'answer'] = df['prerequisites']
    
        return df
