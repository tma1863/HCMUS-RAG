import json
import pandas as pd
import os
import random

class DataframePostprocessor:
    def __init__(self, df: pd.DataFrame, result_dict: dict):
        if df is None:
            raise ValueError("❌ Can't find dataframe for postprocessor")
        if result_dict is None:
            raise ValueError("❌ Can't find result dictionary for postprocessor")

        self.df = df
        self.result_dict = result_dict
        self.merged_df = None

    def overwrite_prerequisites_with_answer(self):
        if 'answer' in self.df.columns:
            self.df['prerequisites'] = self.df['answer']
        else:
            print("⚠️ Column 'answer' does not exist in the DataFrame.")

        for col in ['answer']:
            if col in self.df.columns:
                self.df.drop(columns=col, inplace=True)

    def group_by_course_id(self):
        self.merged_df = self.df.groupby(['course_id', 'course_name'], as_index=False).agg({
            'prerequisites': lambda x: ', '.join(sorted(set(i.strip() for i in x if i.strip())))
        })

    def update_result_dict(self):
        for _, row in self.merged_df.iterrows():
            course_id = row['course_id']
            if course_id in self.result_dict:
                self.result_dict[course_id][4] = row['prerequisites']
            else:
                print(f"⚠️ course_id {course_id} does not exist.")

    def label_and_export_json(self, output_path="output.json"):
        labels = [
            "course name: ",
            "semester: ",
            "teacher name: ",
            "course type: ",
            "required prerequisites: ",
            "learning outcomes: ",
            "content: "
        ]

        labeled_dict = {}
        for key, values in self.result_dict.items():
            labeled_values = [
                label + value for label, value in zip(labels, values)
            ]
            labeled_values.insert(0, "course id: " + key)
            labeled_dict[key] = labeled_values

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(labeled_dict, f, indent=4, ensure_ascii=False)

        print(f"\n-Total number of courses: {len(labeled_dict)}")

    def run(self, output_filename):
        output_path = os.path.join(".\data_processing_pipeline\outputs", output_filename)
        
        self.overwrite_prerequisites_with_answer()
        self.group_by_course_id()
        self.update_result_dict()

        self.label_and_export_json(output_path)
        print("\n✅ Your output file is done! \n")

