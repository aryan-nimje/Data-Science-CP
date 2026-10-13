import os
import joblib
import pandas as pd
import numpy as np


class FeatureImportanceEngine:

    def __init__(
        self,
        model_path="models/best_model_pipeline.pkl"
    ):
        self.model_path = model_path
        self.pipeline = None
        self.preprocessor = None
        self.model = None

    def load_model(self):

        if not os.path.exists(
            self.model_path
        ):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.pipeline = joblib.load(
            self.model_path
        )

        self.preprocessor = (
            self.pipeline.named_steps[
                "preprocessing"
            ]
        )

        self.model = (
            self.pipeline.named_steps[
                "model"
            ]
        )

        return self.pipeline

    def get_feature_names(self):

        feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        return feature_names

    def calculate_importance(self):

        if self.pipeline is None:
            self.load_model()

        feature_names = (
            self.get_feature_names()
        )

        model_name = (
            self.model.__class__.__name__
        )

        # ------------------------------------------
        # LINEAR MODELS
        # ------------------------------------------

        if hasattr(
            self.model,
            "coef_"
        ):

            coefficients = (
                self.model.coef_
            )

            # Binary classification
            if coefficients.ndim == 2:

                importance = np.abs(
                    coefficients[0]
                )

                direction = coefficients[0]

            else:

                importance = np.abs(
                    coefficients
                )

                direction = coefficients

        # ------------------------------------------
        # TREE MODELS
        # ------------------------------------------

        elif hasattr(
            self.model,
            "feature_importances_"
        ):

            importance = (
                self.model.feature_importances_
            )

            direction = (
                importance.copy()
            )

        else:

            raise ValueError(
                f"Feature importance is not "
                f"supported for {model_name}"
            )

        results = []

        for feature, value, direction_value in zip(
            feature_names,
            importance,
            direction
        ):

            results.append(
                {
                    "feature":
                        str(feature),

                    "importance":
                        round(
                            float(value),
                            6
                        ),

                    "direction":
                        (
                            "Positive"
                            if direction_value > 0
                            else
                            "Negative"
                            if direction_value < 0
                            else
                            "Neutral"
                        )
                }
            )

        results.sort(
            key=lambda item:
                item["importance"],
            reverse=True
        )

        total_importance = sum(
            item["importance"]
            for item in results
        )

        for item in results:

            if total_importance > 0:

                item[
                    "importance_percentage"
                ] = round(
                    (
                        item["importance"]
                        /
                        total_importance
                    ) * 100,
                    2
                )

            else:

                item[
                    "importance_percentage"
                ] = 0.0

        return {
            "model":
                model_name,

            "method":
                (
                    "Absolute Coefficients"
                    if hasattr(
                        self.model,
                        "coef_"
                    )
                    else
                    "Tree Feature Importance"
                ),

            "features":
                results
        }

    def get_top_features(
        self,
        top_n=10
    ):

        result = self.calculate_importance()

        return {
            "model":
                result["model"],

            "method":
                result["method"],

            "top_features":
                result["features"][:top_n]
        }