import pandas as pd
import numpy as np


class AutomaticPreprocessor:

    def __init__(self, dataframe, target_column):

        self.df = dataframe.copy()
        self.target_column = target_column

    # ==========================================================
    # IDENTIFY ID-LIKE COLUMNS
    # ==========================================================

    def detect_id_columns(self):

        id_columns = []

        for column in self.df.columns:

            if column == self.target_column:
                continue

            unique_ratio = (
                self.df[column]
                .nunique(dropna=True)
                / len(self.df)
            )

            column_name = column.lower()

            id_keywords = [
                "id",
                "index",
                "uuid",
                "identifier",
                "roll_no",
                "student_no",
                "admission_no"
            ]

            if any(
                keyword in column_name
                for keyword in id_keywords
            ):

                id_columns.append(column)

            elif unique_ratio >= 0.95:

                id_columns.append(column)

        return id_columns

    # ==========================================================
    # IDENTIFY CONSTANT COLUMNS
    # ==========================================================

    def detect_constant_columns(self):

        constant_columns = []

        for column in self.df.columns:

            if column == self.target_column:
                continue

            if (
                self.df[column]
                .nunique(dropna=False)
                <= 1
            ):

                constant_columns.append(column)

        return constant_columns

    # ==========================================================
    # IDENTIFY NUMERICAL FEATURES
    # ==========================================================

    def detect_numerical_features(
        self,
        excluded_columns
    ):

        numerical = (
            self.df
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        return [
            column
            for column in numerical
            if column not in excluded_columns
            and column != self.target_column
        ]

    # ==========================================================
    # IDENTIFY CATEGORICAL FEATURES
    # ==========================================================

    def detect_categorical_features(
        self,
        excluded_columns
    ):

        categorical = (
            self.df
            .select_dtypes(
                include=[
                    "object",
                    "category"
                ]
            )
            .columns
            .tolist()
        )

        return [
            column
            for column in categorical
            if column not in excluded_columns
            and column != self.target_column
        ]

    # ==========================================================
    # MISSING VALUE ANALYSIS
    # ==========================================================

    def analyze_missing_values(
        self,
        numerical_features,
        categorical_features
    ):

        numerical_missing = {}
        categorical_missing = {}

        for column in numerical_features:

            count = int(
                self.df[column]
                .isnull()
                .sum()
            )

            if count > 0:

                numerical_missing[column] = {
                    "count": count,
                    "strategy": "Median Imputation"
                }

        for column in categorical_features:

            count = int(
                self.df[column]
                .isnull()
                .sum()
            )

            if count > 0:

                categorical_missing[column] = {
                    "count": count,
                    "strategy":
                        "Most Frequent Imputation"
                }

        return {
            "numerical": numerical_missing,
            "categorical": categorical_missing
        }

    # ==========================================================
    # CREATE PREPROCESSING PLAN
    # ==========================================================

    def generate_plan(self):

        id_columns = (
            self.detect_id_columns()
        )

        constant_columns = (
            self.detect_constant_columns()
        )

        columns_to_remove = list(
            set(
                id_columns +
                constant_columns
            )
        )

        numerical_features = (
            self.detect_numerical_features(
                columns_to_remove
            )
        )

        categorical_features = (
            self.detect_categorical_features(
                columns_to_remove
            )
        )

        missing_values = (
            self.analyze_missing_values(
                numerical_features,
                categorical_features
            )
        )

        return {

            "target": self.target_column,

            "features": {
                "numerical":
                    numerical_features,

                "categorical":
                    categorical_features
            },

            "removed_columns": {
                "id_like":
                    id_columns,

                "constant":
                    constant_columns
            },

            "missing_values":
                missing_values,

            "transformations": {

                "numerical": [
                    "Median Imputation",
                    "Standard Scaling"
                ],

                "categorical": [
                    "Most Frequent Imputation",
                    "One-Hot Encoding"
                ]
            }
        }