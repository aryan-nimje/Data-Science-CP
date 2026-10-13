import json
import os

from backend.ml_engine.insight_engine import (
    InsightEngine
)


class ModelInvestigationReport:

    def __init__(
        self,
        training_result,
        error_result,
        importance_result,
        shap_result=None,
        target=None,
        problem_type=None
    ):
        self.training_result = training_result
        self.error_result = error_result
        self.importance_result = importance_result
        self.shap_result = shap_result

        self.target = target
        self.problem_type = problem_type

    # ==========================================================
    # MODEL PERFORMANCE
    # ==========================================================

    def get_model_performance(self):

        best_model = self.training_result.get(
            "best_model"
        )

        results = self.training_result.get(
            "results",
            {}
        )

        model_result = results.get(
            best_model,
            {}
        )

        return {
            "model": best_model,
            "metrics": model_result
        }

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    def get_feature_analysis(self):

        features = self.importance_result.get(
            "features",
            []
        )

        return {
            "method":
                self.importance_result.get(
                    "method"
                ),

            "top_features":
                features[:5]
        }

    # ==========================================================
    # ERROR ANALYSIS
    # ==========================================================

    def get_error_analysis(self):

        return {
            "total_rows":
                self.error_result.get(
                    "total_rows"
                ),

            "correct_predictions":
                self.error_result.get(
                    "correct_predictions"
                ),

            "incorrect_predictions":
                self.error_result.get(
                    "incorrect_predictions"
                ),

            "error_percentage":
                self.error_result.get(
                    "error_percentage"
                ),

            "confusion_matrix":
                self.error_result.get(
                    "confusion_matrix"
                ),

            "class_metrics":
                self.error_result.get(
                    "class_metrics"
                ),

            "misclassified_rows":
                self.error_result.get(
                    "misclassified_rows"
                )
        }

    # ==========================================================
    # SHAP ANALYSIS
    # ==========================================================

    def get_shap_analysis(self):

        if not self.shap_result:

            return None

        return {
            "model":
                self.shap_result.get(
                    "model"
                ),

            "base_value":
                self.shap_result.get(
                    "base_value"
                ),

            "positive_factors":
                self.shap_result.get(
                    "positive_factors",
                    []
                )[:5],

            "negative_factors":
                self.shap_result.get(
                    "negative_factors",
                    []
                )[:5]
        }

    # ==========================================================
    # BASE REPORT
    # ==========================================================

    def generate_report_without_insights(self):

        return {

            "report_type":
                "AI Model Investigation Report",

            "target":
                self.target,

            "problem_type":
                self.problem_type,

            "model_performance":
                self.get_model_performance(),

            "feature_analysis":
                self.get_feature_analysis(),

            "error_analysis":
                self.get_error_analysis(),

            "shap_analysis":
                self.get_shap_analysis()
        }

    # ==========================================================
    # AI INSIGHTS
    # ==========================================================

    def generate_ai_insights(
        self,
        base_report,
        what_if_result=None
    ):

        insight_engine = InsightEngine(
            report=base_report,
            what_if_result=what_if_result
        )

        return insight_engine.generate()

    # ==========================================================
    # COMPLETE REPORT
    # ==========================================================

    def generate_report(
        self,
        what_if_result=None
    ):

        # Build the basic report first
        report = (
            self.generate_report_without_insights()
        )

        # Generate AI insights
        ai_insights = (
            self.generate_ai_insights(
                report,
                what_if_result
            )
        )

        # Add insights to report
        report[
            "ai_insights"
        ] = ai_insights

        return report

    # ==========================================================
    # SAVE JSON REPORT
    # ==========================================================

    def save_json(
        self,
        path="reports/model_investigation_report.json",
        what_if_result=None
    ):

        report = self.generate_report(
            what_if_result
        )

        directory = os.path.dirname(
            path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False
            )

        return path