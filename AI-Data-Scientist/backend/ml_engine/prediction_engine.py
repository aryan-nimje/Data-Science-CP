import os
import joblib
import pandas as pd


class PredictionEngine:

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
    # SINGLE PREDICTION
    # ==================================================

    def predict_single(
        self,
        input_data
    ):

        if self.pipeline is None:
            self.load_model()

        # Convert dictionary into DataFrame
        input_df = pd.DataFrame(
            [input_data]
        )

        prediction = (
            self.pipeline.predict(
                input_df
            )
        )

        result = {
            "prediction":
                prediction[0]
        }

        # ----------------------------------------------
        # Probability
        # ----------------------------------------------

        if hasattr(
            self.pipeline,
            "predict_proba"
        ):

            probabilities = (
                self.pipeline.predict_proba(
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

    # ==================================================
    # BATCH PREDICTION
    # ==================================================

    def predict_batch(
        self,
        dataframe
    ):

        if self.pipeline is None:
            self.load_model()

        predictions = (
            self.pipeline.predict(
                dataframe
            )
        )

        result = dataframe.copy()

        result[
            "Prediction"
        ] = predictions

        # ----------------------------------------------
        # Add probabilities
        # ----------------------------------------------

        if hasattr(
            self.pipeline,
            "predict_proba"
        ):

            probabilities = (
                self.pipeline.predict_proba(
                    dataframe
                )
            )

            result[
                "Confidence"
            ] = (
                probabilities.max(
                    axis=1
                ) * 100
            ).round(2)

        return result