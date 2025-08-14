import time
from data_processing_pipeline.preprocessing.table_extractor import TableExtractor
from data_processing_pipeline.preprocessing.table_cleaner import TableCleaner
from data_processing_pipeline.preprocessing.course_cleaner import CourseCleaner
from data_processing_pipeline.preprocessing.prerequisite_extractor import PrerequisiteExtractor
from data_processing_pipeline.preprocessing.file_saver import FileSaver
from data_processing_pipeline.preprocessing.df_postprocessor import DataframePostprocessor

class PreprocessingPipeline:
    def __init__(self, filepath, module_name):
        self.filepath = filepath
        self.module_name = module_name
        self.extractor = None
        self.tables = None
        self.filtered = None
        self.df = None
        self.result_dict = None

    def run(self):
        self.extract_tables()
        self.clean_tables()

        start_time = time.time()
        self.clean_course_info()
        self.extract_prerequisites()
        self.save_for_manual_edit()
        self.run_postprocessor()
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n Execution time: {elapsed:.2f}s")

    def extract_tables(self):
        self.extractor = TableExtractor(self.filepath)
        self.tables = self.extractor.extract_tables_with_headings()

    def clean_tables(self):
        cleaner = TableCleaner(self.tables)
        remove_keys = [
            "Language",
            "Teaching methods",
            "Workload (incl. contact hours, self-study hours)",
            "Credit points",
            "Examination forms",
            "Study and examination requirements",
            "Reading list"
        ]
        self.filtered = cleaner.remove_unwanted_rows(remove_keys=remove_keys)
        cleaner.print_invalid_tables()
        cleaner.run_deletion_loop()

    def clean_course_info(self):
        course_cleaner = CourseCleaner(self.filtered)
        course_cleaner.clean_course_ids()
        course_cleaner.normalize_semesters()
        course_cleaner.normalize_course_type()
        course_cleaner.clean_teacher_names()
        course_cleaner.clean_course_titles_and_prerequisites()

    def extract_prerequisites(self):
        prerequisite_extractor = PrerequisiteExtractor(self.filtered)
        prerequisite_extractor.create_mappings()
        self.df = prerequisite_extractor.extract_dataframe()
        self.df = prerequisite_extractor.split_prerequisites(self.df)
        self.df = prerequisite_extractor.standardize_roman_numerals(self.df)
        self.df = prerequisite_extractor.match_fuzzy(self.df)
        self.df = PrerequisiteExtractor.adjust_answer_column(self.df)
        self.result_dict = prerequisite_extractor.result_dict

    def save_for_manual_edit(self):
        saver = FileSaver()
        saver.download(
            self.df,
            filename=f"{self.module_name}.csv",
            output_dir=".\data_processing_pipeline\logs",
            result_dict=self.result_dict
        )
        

    def run_postprocessor(self):
        postprocessor = DataframePostprocessor(self.df, self.result_dict)
        postprocessor.run(f"{self.module_name}_output.json")
