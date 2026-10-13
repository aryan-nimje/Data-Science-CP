import pandas as pd
import numpy as np


class DataQualityAnalyzer:

    def __init__(self, dataframe):
        self.df = dataframe

    # --------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------

    def analyze_missing_values(self):

        result = {}

        for column in self.df.columns:

            missing_count = int(self.df[column].isnull().sum())

            if missing_count > 0:

                percentage = round(
                    (missing_count / len(self.df)) * 100,
                    2
                )

                result[column] = {
                    "missing_count": missing_count,
                    "missing_percentage": percentage
                }

        return result

    # --------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------

    def analyze_duplicates(self):

        duplicate_count = int(
            self.df.duplicated().sum()
        )

        duplicate_percentage = round(
            (duplicate_count / len(self.df)) * 100,
            2
        )

        return {
            "count": duplicate_count,
            "percentage": duplicate_percentage
        }

    # --------------------------------------------------
    # CONSTANT COLUMNS
    # --------------------------------------------------

    def detect_constant_columns(self):

        constant_columns = []

        for column in self.df.columns:

            if self.df[column].nunique(
                dropna=False
            ) <= 1:

                constant_columns.append(column)

        return constant_columns

    # --------------------------------------------------
    # HIGH CARDINALITY
    # --------------------------------------------------

    def detect_high_cardinality(self):

        high_cardinality = {}

        for column in self.df.select_dtypes(
            include=["object", "category"]
        ).columns:

            unique_count = self.df[column].nunique()

            unique_percentage = (
                unique_count / len(self.df)
            ) * 100

            if unique_percentage >= 50:

                high_cardinality[column] = {
                    "unique_values": int(unique_count),
                    "percentage": round(
                        unique_percentage,
                        2
                    )
                }

        return high_cardinality

    # --------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------

    def detect_class_imbalance(self, target_column):

        if target_column not in self.df.columns:
            return {
                "error": "Target column not found"
            }

        target = self.df[target_column]

        class_distribution = (
            target.value_counts(normalize=True) * 100
        ).round(2)

        imbalance_detected = False

        if len(class_distribution) > 1:

            largest = class_distribution.max()
            smallest = class_distribution.min()

            if largest >= 70 and smallest <= 30:
                imbalance_detected = True

        return {
            "distribution":
                class_distribution.to_dict(),

            "imbalance_detected":
                imbalance_detected
        }

    # --------------------------------------------------
    # DATASET HEALTH SCORE
    # --------------------------------------------------

    def calculate_health_score(self):

        score = 100

        # Missing values
        total_missing = self.df.isnull().sum().sum()

        if total_missing > 0:

            missing_percentage = (
                total_missing /
                (self.df.shape[0] * self.df.shape[1])
            ) * 100

            score -= min(
                missing_percentage * 2,
                25
            )

        # Duplicate rows
        duplicates = self.df.duplicated().sum()

        duplicate_percentage = (
            duplicates / len(self.df)
        ) * 100

        score -= min(
            duplicate_percentage * 1.5,
            15
        )

        # Constant columns
        constant_columns = self.detect_constant_columns()

        score -= min(
            len(constant_columns) * 5,
            20
        )

        # Keep score between 0 and 100
        score = max(
            0,
            min(100, score)
        )

        return round(score, 2)

    # --------------------------------------------------
    # COMPLETE ANALYSIS
    # --------------------------------------------------

    def analyze(self):

        return {

            "missing_values":
                self.analyze_missing_values(),

            "duplicates":
                self.analyze_duplicates(),

            "constant_columns":
                self.detect_constant_columns(),

            "high_cardinality":
                self.detect_high_cardinality(),

            "health_score":
                self.calculate_health_score()
        }