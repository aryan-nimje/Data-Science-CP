import os
import joblib
import pandas as pd


class WhatIfEngine:

    def __init__(
        self,
        model_path="models/best_model_pipeline.pkl"
    ):
        self.model_path = model_path
        self.pipeline = None

    # ==========================================================
    # LOAD MODEL
    # ==========================================================

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

        return self.pipeline

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(self, input_data):

        if self.pipeline is None:
            self.load_model()

        input_df = pd.DataFrame(
            [input_data]
        )

        prediction = (
            self.pipeline.predict(
                input_df
            )
        )[0]

        result = {
            "prediction":
                prediction
        }

        # Classification probability
        if hasattr(
            self.pipeline,
            "predict_proba"
        ):

            probabilities = (
                self.pipeline
                .predict_proba(
                    input_df
                )[0]
            )

            classes = (
                self.pipeline.classes_
            )

            probability_data = {}

            for class_name, probability in zip(
                classes,
                probabilities
            ):

                probability_data[
                    str(class_name)
                ] = round(
                    float(probability) * 100,
                    2
                )

            result[
                "class_probabilities"
            ] = probability_data

            result[
                "confidence"
            ] = round(
                float(
                    max(probabilities)
                ) * 100,
                2
            )

        return result

    # ==========================================================
    # COMPARE TWO SCENARIOS
    # ==========================================================

    def compare(
        self,
        original_data,
        modified_data
    ):

        original_result = (
            self.predict(
                original_data
            )
        )

        modified_result = (
            self.predict(
                modified_data
            )
        )

        changed_features = []

        all_features = set(
            original_data.keys()
        ).union(
            modified_data.keys()
        )

        for feature in all_features:

            original_value = (
                original_data.get(
                    feature
                )
            )

            modified_value = (
                modified_data.get(
                    feature
                )
            )

            if original_value != modified_value:

                changed_features.append(
                    {
                        "feature":
                            feature,

                        "original_value":
                            original_value,

                        "modified_value":
                            modified_value
                    }
                )

        prediction_changed = (
            original_result[
                "prediction"
            ]
            !=
            modified_result[
                "prediction"
            ]
        )

        confidence_change = None

        if (
            "confidence"
            in original_result
            and
            "confidence"
            in modified_result
        ):

            confidence_change = round(
                modified_result[
                    "confidence"
                ]
                -
                original_result[
                    "confidence"
                ],
                2
            )

        return {

            "original": {
                "input":
                    original_data,

                "prediction":
                    original_result[
                        "prediction"
                    ],

                "confidence":
                    original_result.get(
                        "confidence"
                    ),

                "class_probabilities":
                    original_result.get(
                        "class_probabilities"
                    )
            },

            "modified": {
                "input":
                    modified_data,

                "prediction":
                    modified_result[
                        "prediction"
                    ],

                "confidence":
                    modified_result.get(
                        "confidence"
                    ),

                "class_probabilities":
                    modified_result.get(
                        "class_probabilities"
                    )
            },

            "changed_features":
                changed_features,

            "prediction_changed":
                prediction_changed,

            "confidence_change":
                confidence_change
        }