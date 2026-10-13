import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ErrorAnalyzer:

    def __init__(
        self,
        model_path="models/best_model_pipeline.pkl"
    ):
        self.model_path = model_path
        self.pipeline = None

    # ==================================================
    # LOAD MODEL
    # ==================================================

    def load_model(self):

        if not os.path.exists(
            self.model_path
        ):
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        self.pipeline = joblib.load(
            self.model_path
        )

        return self.pipeline

    # ==================================================
    # CLASSIFICATION ANALYSIS
    # ==================================================

    def analyze_classification(
        self,
        dataframe,
        target_column
    ):

        if self.pipeline is None:
            self.load_model()

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                f"not found."
            )

        # ----------------------------------------------
        # Separate features and target
        # ----------------------------------------------

        X = dataframe.drop(
            columns=[target_column]
        )

        y = dataframe[
            target_column
        ]

        # ----------------------------------------------
        # Predictions
        # ----------------------------------------------

        predictions = (
            self.pipeline.predict(X)
        )

        # ----------------------------------------------
        # Basic metrics
        # ----------------------------------------------

        accuracy = accuracy_score(
            y,
            predictions
        )

        precision = precision_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )

        # ----------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------

        labels = sorted(
            y.astype(str).unique()
        )

        matrix = confusion_matrix(
            y.astype(str),
            pd.Series(
                predictions
            ).astype(str),
            labels=labels
        )

        # ----------------------------------------------
        # Classification Report
        # ----------------------------------------------

        report = classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0
        )

        # ----------------------------------------------
        # Correct / incorrect predictions
        # ----------------------------------------------

        correct_mask = (
            y.astype(str).values
            ==
            pd.Series(
                predictions
            ).astype(str).values
        )

        correct_count = int(
            correct_mask.sum()
        )

        incorrect_count = int(
            (~correct_mask).sum()
        )

        total = len(y)

        error_percentage = (
            incorrect_count / total * 100
            if total > 0
            else 0
        )

        # ----------------------------------------------
        # Misclassified rows
        # ----------------------------------------------

        misclassified = dataframe[
            ~correct_mask
        ].copy()

        misclassified[
            "Actual"
        ] = y[
            ~correct_mask
        ].values

        misclassified[
            "Predicted"
        ] = predictions[
            ~correct_mask
        ]

        # Convert to JSON-friendly records
        misclassified_records = (
            misclassified
            .reset_index()
            .rename(
                columns={
                    "index": "Row"
                }
            )
            .to_dict(
                orient="records"
            )
        )

        # ----------------------------------------------
        # Confusion matrix as JSON
        # ----------------------------------------------

        confusion_matrix_data = []

        for row in matrix:

            confusion_matrix_data.append(
                [
                    int(value)
                    for value in row
                ]
            )

        # ----------------------------------------------
        # Class metrics
        # ----------------------------------------------

        class_metrics = {}

        for label in labels:

            label_key = str(label)

            if label_key in report:

                class_metrics[
                    label_key
                ] = {
                    "precision":
                        round(
                            report[
                                label_key
                            ]["precision"],
                            4
                        ),

                    "recall":
                        round(
                            report[
                                label_key
                            ]["recall"],
                            4
                        ),

                    "f1_score":
                        round(
                            report[
                                label_key
                            ]["f1-score"],
                            4
                        ),

                    "support":
                        int(
                            report[
                                label_key
                            ]["support"]
                        )
                }

        return {

            "problem_type":
                "Classification",

            "total_rows":
                total,

            "correct_predictions":
                correct_count,

            "incorrect_predictions":
                incorrect_count,

            "error_percentage":
                round(
                    error_percentage,
                    2
                ),

            "metrics": {

                "accuracy":
                    round(
                        accuracy,
                        4
                    ),

                "precision":
                    round(
                        precision,
                        4
                    ),

                "recall":
                    round(
                        recall,
                        4
                    ),

                "f1_score":
                    round(
                        f1,
                        4
                    )
            },

            "confusion_matrix": {

                "labels":
                    labels,

                "matrix":
                    confusion_matrix_data
            },

            "class_metrics":
                class_metrics,

            "misclassified_rows":
                misclassified_records
        }

    # ==================================================
    # REGRESSION ANALYSIS
    # ==================================================

    def analyze_regression(
        self,
        dataframe,
        target_column
    ):

        if self.pipeline is None:
            self.load_model()

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                f"not found."
            )

        X = dataframe.drop(
            columns=[target_column]
        )

        y = dataframe[
            target_column
        ]

        predictions = (
            self.pipeline.predict(X)
        )

        mae = mean_absolute_error(
            y,
            predictions
        )

        mse = mean_squared_error(
            y,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y,
            predictions
        )

        errors = (
            y.values
            -
            predictions
        )

        error_dataframe = dataframe.copy()

        error_dataframe[
            "Actual"
        ] = y.values

        error_dataframe[
            "Predicted"
        ] = predictions

        error_dataframe[
            "Absolute_Error"
        ] = np.abs(errors)

        error_dataframe[
            "Percentage_Error"
        ] = np.where(
            y.values != 0,
            (
                np.abs(errors)
                /
                np.abs(y.values)
            ) * 100,
            0
        )

        # Largest errors first
        error_dataframe = (
            error_dataframe
            .sort_values(
                "Absolute_Error",
                ascending=False
            )
        )

        return {

            "problem_type":
                "Regression",

            "total_rows":
                len(y),

            "metrics": {

                "mae":
                    round(
                        float(mae),
                        4
                    ),

                "mse":
                    round(
                        float(mse),
                        4
                    ),

                "rmse":
                    round(
                        float(rmse),
                        4
                    ),

                "r2":
                    round(
                        float(r2),
                        4
                    )
            },

            "largest_errors":
                error_dataframe
                .head(10)
                .reset_index()
                .rename(
                    columns={
                        "index": "Row"
                    }
                )
                .to_dict(
                    orient="records"
                )
        }

    # ==================================================
    # AUTOMATIC PROBLEM-TYPE ANALYSIS
    # ==================================================

    def analyze(
        self,
        dataframe,
        target_column,
        problem_type
    ):

        if "Classification" in problem_type:

            return self.analyze_classification(
                dataframe,
                target_column
            )

        elif problem_type == "Regression":

            return self.analyze_regression(
                dataframe,
                target_column
            )

        else:

            raise ValueError(
                f"Unsupported problem type: "
                f"{problem_type}"
            )