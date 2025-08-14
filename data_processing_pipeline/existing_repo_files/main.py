from data_processing_pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline
import re

if __name__ == "__main__":
    filepath = input("\n Please enter the file path: ")
    match = re.search(r'Module-Handbook_(.*?)_', filepath)
    if match:
        module_name = match.group(1)
        pipeline = PreprocessingPipeline(filepath, module_name)
        pipeline.run()