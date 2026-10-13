import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from scipy.stats import zscore


class OutlierDetector:

    def __init__(self, dataframe):
        self.df = dataframe

    # --------------------------------------------------
    # IQR METHOD
    # --------------------------------------------------

    def detect_iqr(self):

        results = {}

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for column in numerical_columns:

            data = self.df[column].dropna()

            if len(data) < 4:
                continue

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (
                (data < lower_bound) |
                (data > upper_bound)
            )

            outlier_count = int(outlier_mask.sum())

            results[column] = {
                "method": "IQR",
                "Q1": round(float(Q1), 4),
                "Q3": round(float(Q3), 4),
                "lower_bound": round(
                    float(lower_bound), 4
                ),
                "upper_bound": round(
                    float(upper_bound), 4
                ),
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    (outlier_count / len(data)) * 100,
                    2
                )
            }

        return results

    # --------------------------------------------------
    # Z-SCORE METHOD
    # --------------------------------------------------

    def detect_zscore(self, threshold=3):

        results = {}

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for column in numerical_columns:

            data = self.df[column].dropna()

            if len(data) < 4:
                continue

            scores = np.abs(zscore(data))

            outlier_mask = scores > threshold

            outlier_count = int(
                np.sum(outlier_mask)
            )

            results[column] = {
                "method": "Z-Score",
                "threshold": threshold,
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    (outlier_count / len(data)) * 100,
                    2
                )
            }

        return results

    # --------------------------------------------------
    # ISOLATION FOREST
    # --------------------------------------------------

    def detect_isolation_forest(
        self,
        contamination=0.05
    ):

        results = {}

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not numerical_columns:
            return results

        data = self.df[numerical_columns].copy()

        # Replace missing values with median
        data = data.fillna(
            data.median(numeric_only=True)
        )

        if len(data) < 10:
            return {
                "error":
                    "Isolation Forest requires "
                    "at least 10 rows."
            }

        model = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        predictions = model.fit_predict(data)

        anomaly_mask = predictions == -1

        anomaly_count = int(
            anomaly_mask.sum()
        )

        results["dataset"] = {
            "method": "Isolation Forest",
            "contamination": contamination,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round(
                (anomaly_count / len(data)) * 100,
                2
            )
        }

        # Store suspicious row indexes
        anomaly_indexes = (
            data.index[anomaly_mask]
            .tolist()
        )

        results["dataset"]["row_indexes"] = [
            int(index)
            for index in anomaly_indexes
        ]

        return results

    # --------------------------------------------------
    # COMPLETE OUTLIER ANALYSIS
    # --------------------------------------------------

    def analyze(self):

        return {
            "iqr": self.detect_iqr(),
            "zscore": self.detect_zscore(),
            "isolation_forest":
                self.detect_isolation_forest()
        }