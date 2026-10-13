from backend.data_engine.profiler import DatasetProfiler
from backend.data_engine.quality import DataQualityAnalyzer
from backend.data_engine.outliers import OutlierDetector


class DatasetInvestigator:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def investigate(self):

        # ---------------------------------------------
        # 1. PROFILE
        # ---------------------------------------------

        profiler = DatasetProfiler(self.file_path)

        self.df = profiler.load_dataset()

        profile = profiler.get_basic_info()

        # ---------------------------------------------
        # 2. DATA QUALITY
        # ---------------------------------------------

        quality_analyzer = DataQualityAnalyzer(self.df)

        quality_report = quality_analyzer.analyze()

        # ---------------------------------------------
        # 3. OUTLIERS
        # ---------------------------------------------

        outlier_detector = OutlierDetector(self.df)

        outlier_report = outlier_detector.analyze()

        # ---------------------------------------------
        # 4. COMBINE EVERYTHING
        # ---------------------------------------------

        report = {
            "dataset": {
                "file": self.file_path,
                "rows": profile["rows"],
                "columns": profile["columns"]
            },

            "structure": {
                "numerical_columns":
                    profile["numerical_columns"],

                "categorical_columns":
                    profile["categorical_columns"],

                "column_types":
                    profile["column_types"],

                "unique_values":
                    profile["unique_values"]
            },

            "quality": quality_report,

            "outliers": outlier_report
        }

        return report