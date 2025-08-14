import os
import json
import pandas as pd

class FileSaver:
    def download(self, df: pd.DataFrame, filename="check_dataframe.csv", output_dir=".", result_dict: dict = None):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"\n✅ File saved to: {path} \n")

        if result_dict is not None:
            json_filename = filename.replace(".csv", "_result_dict.json")
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
            print(f"\n✅ result_dict saved to: {json_path} \n")