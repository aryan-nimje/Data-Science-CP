import pandas as pd
import numpy as np


class DatasetProfiler:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_dataset(self):
        """Load CSV dataset."""

        self.df = pd.read_csv(self.file_path)

        return self.df

    def get_basic_info(self):
        """Generate basic dataset information."""

        if self.df is None:
            self.load_dataset()

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = self.df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        missing_values = self.df.isnull().sum()

        missing_percentage = (
            missing_values / len(self.df) * 100
        ).round(2)

        profile = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],

            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,

            "missing_values": missing_values[
                missing_values > 0
            ].to_dict(),

            "missing_percentage": missing_percentage[
                missing_percentage > 0
            ].to_dict(),

            "duplicate_rows": int(
                self.df.duplicated().sum()
            ),

            "column_types": self.df.dtypes.astype(
                str
            ).to_dict(),

            "unique_values": self.df.nunique().to_dict()
        }

        return profile

    def get_statistics(self):
        """Generate statistical summary."""

        if self.df is None:
            self.load_dataset()

        return self.df.describe(
            include="all"
        ).transpose()