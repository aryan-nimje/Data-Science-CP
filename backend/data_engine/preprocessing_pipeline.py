import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


class PreprocessingPipeline:

    def __init__(
        self,
        dataframe,
        target_column,
        numerical_features,
        categorical_features,
        removed_columns
    ):

        self.df = dataframe.copy()

        self.target_column = target_column

        self.numerical_features = (
            numerical_features
        )

        self.categorical_features = (
            categorical_features
        )

        self.removed_columns = (
            removed_columns
        )

        self.pipeline = None

    # ==========================================================
    # BUILD PIPELINE
    # ==========================================================

    def build(self):

        # ------------------------------------------------------
        # Numerical processing
        # ------------------------------------------------------

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # ------------------------------------------------------
        # Categorical processing
        # ------------------------------------------------------

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        # ------------------------------------------------------
        # Combine pipelines
        # ------------------------------------------------------

        self.pipeline = ColumnTransformer(

            transformers=[

                (
                    "numerical",
                    numerical_pipeline,
                    self.numerical_features
                ),

                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_features
                )
            ],

            remainder="drop"
        )

        return self.pipeline

    # ==========================================================
    # PREPARE FEATURES AND TARGET
    # ==========================================================

    def prepare_data(self):

        columns_to_remove = (
            self.removed_columns
        )

        X = self.df.drop(
            columns=[
                self.target_column
            ] + columns_to_remove,
            errors="ignore"
        )

        y = self.df[
            self.target_column
        ]

        return X, y

    # ==========================================================
    # FIT AND TRANSFORM
    # ==========================================================

    def fit_transform(self):

        if self.pipeline is None:

            self.build()

        X, y = self.prepare_data()

        X_transformed = (
            self.pipeline.fit_transform(X)
        )

        return X_transformed, y

    # ==========================================================
    # GET FEATURE NAMES
    # ==========================================================

    def get_feature_names(self):

        if self.pipeline is None:

            raise ValueError(
                "Pipeline has not been fitted yet."
            )

        feature_names = (
            self.pipeline
            .get_feature_names_out()
        )

        return feature_names.tolist()