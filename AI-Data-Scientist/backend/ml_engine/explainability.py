import os
import joblib
import pandas as pd
import numpy as np
import shap


class ExplainabilityEngine:

    def __init__(
        self,
        model_path="models/best_model_pipeline.pkl",
        background_path="datasets/test.csv"
    ):
        self.model_path = model_path
        self.background_path = background_path

        self.pipeline = None
        self.preprocessor = None
        self.model = None

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

    # ==================================================
    # LOAD BACKGROUND DATA
    # ==================================================

    def load_background_data(
        self,
        input_columns
    ):

        if not os.path.exists(
            self.background_path
        ):
            raise FileNotFoundError(
                f"Background dataset not found: "
                f"{self.background_path}"
            )

        background_df = pd.read_csv(
            self.background_path
        )

        # The background dataset contains the target.
        # Remove it because SHAP explains features.
        target_column = None

        if hasattr(
            self.preprocessor,
            "transformers_"
        ):

            # Target is not part of preprocessing,
            # so infer it from columns that aren't
            # expected model features.
            pass

        # Keep only columns that are supplied
        # to the prediction model.
        background_df = background_df[
            [
                column
                for column in input_columns
                if column in background_df.columns
            ]
        ]

        return background_df

    # ==================================================
    # TRANSFORM DATA
    # ==================================================

    def transform_data(
        self,
        dataframe
    ):

        transformed = (
            self.preprocessor.transform(
                dataframe
            )
        )

        feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            transformed,
            "toarray"
        ):
            transformed = (
                transformed.toarray()
            )

        return transformed, feature_names

    # ==================================================
    # EXPLAIN SINGLE PREDICTION
    # ==================================================

    def explain_prediction(
        self,
        input_data
    ):

        if self.pipeline is None:
            self.load_model()

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------
        # Load representative background
        # ----------------------------------------------

        background_df = (
            self.load_background_data(
                input_df.columns.tolist()
            )
        )

        if background_df.empty:
            raise ValueError(
                "No usable background data found."
            )

        # ----------------------------------------------
        # Transform background + input
        # ----------------------------------------------

        background_transformed, feature_names = (
            self.transform_data(
                background_df
            )
        )

        input_transformed, _ = (
            self.transform_data(
                input_df
            )
        )

        # ==============================================
        # CREATE SHAP EXPLAINER
        # ==============================================

        model_name = (
            self.model.__class__.__name__
        )

        if model_name in [
            "LogisticRegression",
            "LinearRegression"
        ]:

            explainer = (
                shap.LinearExplainer(
                    self.model,
                    background_transformed
                )
            )

        elif model_name in [
            "RandomForestClassifier",
            "RandomForestRegressor",
            "GradientBoostingClassifier",
            "GradientBoostingRegressor"
        ]:

            explainer = (
                shap.TreeExplainer(
                    self.model
                )
            )

        else:

            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        # ==============================================
        # CALCULATE SHAP VALUES
        # ==============================================

        shap_result = explainer(
            input_transformed
        )

        values = shap_result.values

        # Handle different SHAP output shapes
        if values.ndim == 3:

            # Usually:
            # samples × features × classes
            values = values[
                0,
                :,
                1
            ]

        elif values.ndim == 2:

            values = values[0]

        elif values.ndim == 1:

            values = values

        else:

            raise ValueError(
                f"Unexpected SHAP output shape: "
                f"{values.shape}"
            )

        # ==============================================
        # FEATURE CONTRIBUTIONS
        # ==============================================

        contributions = []

        for feature, value in zip(
            feature_names,
            values
        ):

            numeric_value = float(
                value
            )

            # Ignore microscopic floating-point
            # values such as -0.00000000001
            if abs(numeric_value) < 1e-8:
                numeric_value = 0.0

            contributions.append(
                {
                    "feature":
                        str(feature),

                    "shap_value":
                        round(
                            numeric_value,
                            6
                        ),

                    "absolute_impact":
                        round(
                            abs(numeric_value),
                            6
                        )
                }
            )

        contributions.sort(
            key=lambda item:
                item["absolute_impact"],
            reverse=True
        )

        # ==============================================
        # POSITIVE / NEGATIVE FACTORS
        # ==============================================

        positive = [
            item
            for item in contributions
            if item["shap_value"] > 0
        ]

        negative = [
            item
            for item in contributions
            if item["shap_value"] < 0
        ]

        # ==============================================
        # BASE VALUE
        # ==============================================

        base_value = shap_result.base_values

        if np.ndim(base_value) > 0:
            base_value = np.asarray(
                base_value
            ).flatten()[0]

        return {
            "model":
                model_name,

            "base_value":
                round(
                    float(base_value),
                    6
                ),

            "features":
                contributions,

            "positive_factors":
                positive,

            "negative_factors":
                negative
        }