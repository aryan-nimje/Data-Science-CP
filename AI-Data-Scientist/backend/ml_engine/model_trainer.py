import os
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_validate
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelTrainer:

    def __init__(
        self,
        preprocessing_pipeline,
        X,
        y,
        problem_type
    ):
        self.preprocessing_pipeline = (
            preprocessing_pipeline
        )

        self.X = X
        self.y = y
        self.problem_type = problem_type

        self.models = {}
        self.results = {}

        self.best_model = None
        self.best_model_name = None

    # ==================================================
    # MODEL DEFINITIONS
    # ==================================================

    def get_models(self):

        if "Classification" in self.problem_type:

            return {
                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    ),

                "Gradient Boosting":
                    GradientBoostingClassifier(
                        random_state=42
                    )
            }

        elif self.problem_type == "Regression":

            return {
                "Linear Regression":
                    LinearRegression(),

                "Random Forest":
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42
                    ),

                "Gradient Boosting":
                    GradientBoostingRegressor(
                        random_state=42
                    )
            }

        else:
            raise ValueError(
                f"Unsupported problem type: "
                f"{self.problem_type}"
            )

    # ==================================================
    # CREATE COMPLETE ML PIPELINE
    # ==================================================

    def create_pipeline(self, model):

        preprocessing = clone(
            self.preprocessing_pipeline
        )

        complete_pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    preprocessing
                ),
                (
                    "model",
                    clone(model)
                )
            ]
        )

        return complete_pipeline

    # ==================================================
    # CLASSIFICATION CROSS VALIDATION
    # ==================================================

    def cross_validate_classification(
        self,
        model
    ):

        class_counts = self.y.value_counts()

        min_class_count = int(
            class_counts.min()
        )

        # Small datasets may not support 5 folds
        n_splits = min(
            5,
            min_class_count
        )

        if n_splits < 2:

            return {
                "cv_available": False,
                "cv_reason":
                    "Not enough samples per class "
                    "for cross-validation."
            }

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        pipeline = self.create_pipeline(
            model
        )

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted"
        }

        scores = cross_validate(
            pipeline,
            self.X,
            self.y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        return {
            "cv_available": True,
            "cv_folds": n_splits,

            "accuracy_mean":
                round(
                    scores[
                        "test_accuracy"
                    ].mean(),
                    4
                ),

            "accuracy_std":
                round(
                    scores[
                        "test_accuracy"
                    ].std(),
                    4
                ),

            "precision_mean":
                round(
                    scores[
                        "test_precision"
                    ].mean(),
                    4
                ),

            "recall_mean":
                round(
                    scores[
                        "test_recall"
                    ].mean(),
                    4
                ),

            "f1_mean":
                round(
                    scores[
                        "test_f1"
                    ].mean(),
                    4
                ),

            "f1_std":
                round(
                    scores[
                        "test_f1"
                    ].std(),
                    4
                )
        }

    # ==================================================
    # REGRESSION CROSS VALIDATION
    # ==================================================

    def cross_validate_regression(
        self,
        model
    ):

        if len(self.y) < 2:

            return {
                "cv_available": False,
                "cv_reason":
                    "Not enough samples for "
                    "cross-validation."
            }

        n_splits = min(
            5,
            len(self.y)
        )

        if n_splits < 2:

            return {
                "cv_available": False,
                "cv_reason":
                    "Not enough samples for "
                    "cross-validation."
            }

        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        pipeline = self.create_pipeline(
            model
        )

        scoring = {
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2"
        }

        scores = cross_validate(
            pipeline,
            self.X,
            self.y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        mae = -scores["test_mae"]
        mse = -scores["test_mse"]
        rmse = mse ** 0.5

        return {
            "cv_available": True,
            "cv_folds": n_splits,

            "mae_mean":
                round(
                    mae.mean(),
                    4
                ),

            "mae_std":
                round(
                    mae.std(),
                    4
                ),

            "rmse_mean":
                round(
                    rmse.mean(),
                    4
                ),

            "r2_mean":
                round(
                    scores[
                        "test_r2"
                    ].mean(),
                    4
                ),

            "r2_std":
                round(
                    scores[
                        "test_r2"
                    ].std(),
                    4
                )
        }

    # ==================================================
    # FINAL HOLDOUT EVALUATION
    # ==================================================

    def final_holdout_evaluation(
        self,
        model
    ):

        if "Classification" in self.problem_type:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    self.X,
                    self.y,
                    test_size=0.2,
                    random_state=42,
                    stratify=self.y
                )
            )

        else:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    self.X,
                    self.y,
                    test_size=0.2,
                    random_state=42
                )
            )

        pipeline = self.create_pipeline(
            model
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        if "Classification" in self.problem_type:

            return {
                "holdout_accuracy":
                    round(
                        accuracy_score(
                            y_test,
                            predictions
                        ),
                        4
                    ),

                "holdout_f1":
                    round(
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        ),
                        4
                    )
            }

        mse = mean_squared_error(
            y_test,
            predictions
        )

        return {
            "holdout_mae":
                round(
                    mean_absolute_error(
                        y_test,
                        predictions
                    ),
                    4
                ),

            "holdout_rmse":
                round(
                    mse ** 0.5,
                    4
                ),

            "holdout_r2":
                round(
                    r2_score(
                        y_test,
                        predictions
                    ),
                    4
                )
        }

    # ==================================================
    # BEST MODEL SELECTION
    # ==================================================

    def select_best_model(self):

        available_results = {
            name: result
            for name, result
            in self.results.items()
            if result.get(
                "cv_available",
                False
            )
        }

        if not available_results:

            # Fall back to holdout
            available_results = (
                self.results
            )

        if "Classification" in self.problem_type:

            self.best_model_name = max(
                available_results,
                key=lambda name:
                    available_results[name].get(
                        "f1_mean",
                        available_results[name].get(
                            "holdout_f1",
                            0
                        )
                    )
            )

        else:

            self.best_model_name = max(
                available_results,
                key=lambda name:
                    available_results[name].get(
                        "r2_mean",
                        available_results[name].get(
                            "holdout_r2",
                            float("-inf")
                        )
                    )
            )

        return self.best_model_name

    # ==================================================
    # TRAIN ALL MODELS
    # ==================================================

    def train(self):

        print("\n")
        print("=" * 70)
        print(
            "       AUTOMATED MODEL TRAINING + VALIDATION"
        )
        print("=" * 70)

        print(
            f"\nProblem Type: "
            f"{self.problem_type}"
        )

        print(
            "\nData leakage protection: ENABLED"
        )

        print(
            "Preprocessing is fitted separately "
            "inside each validation fold."
        )

        self.models = self.get_models()

        for name, model in self.models.items():

            print(
                f"\nEvaluating: {name}"
            )

            if "Classification" in (
                self.problem_type
            ):

                cv_results = (
                    self.cross_validate_classification(
                        model
                    )
                )

            else:

                cv_results = (
                    self.cross_validate_regression(
                        model
                    )
                )

            holdout_results = (
                self.final_holdout_evaluation(
                    model
                )
            )

            self.results[name] = {
                **cv_results,
                **holdout_results
            }

        best_name = (
            self.select_best_model()
        )

        # ==========================================
        # TRAIN COMPLETE PIPELINE ON ALL DATA
        # ==========================================

        self.best_model = (
            self.create_pipeline(
                self.models[best_name]
            )
        )

        self.best_model.fit(
            self.X,
            self.y
        )

        print("\n")
        print("=" * 70)
        print(
            "                 MODEL COMPARISON"
        )
        print("=" * 70)

        for name, metrics in (
            self.results.items()
        ):

            print(
                f"\n{name}"
            )

            for metric, value in (
                metrics.items()
            ):

                print(
                    f"  {metric}: {value}"
                )

        print("\n")
        print(
            f"SELECTED MODEL: "
            f"{best_name}"
        )

        print(
            "\n✓ Complete preprocessing + model "
            "pipeline trained on all data"
        )

        print("=" * 70)

        return {
            "results": self.results,
            "best_model": best_name
        }

    # ==================================================
    # SAVE COMPLETE PIPELINE
    # ==================================================

    def save_model(
        self,
        directory="models"
    ):

        if self.best_model is None:

            raise ValueError(
                "Train the models first."
            )

        os.makedirs(
            directory,
            exist_ok=True
        )

        path = os.path.join(
            directory,
            "best_model_pipeline.pkl"
        )

        joblib.dump(
            self.best_model,
            path
        )

        print(
            f"\n✓ Complete pipeline saved to: "
            f"{path}"
        )

        return path