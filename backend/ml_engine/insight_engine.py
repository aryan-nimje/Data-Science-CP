class InsightEngine:

    def __init__(
        self,
        report=None,
        what_if_result=None
    ):
        self.report = report
        self.what_if_result = what_if_result

    # ==========================================================
    # MODEL INSIGHTS
    # ==========================================================

    def model_insights(self):

        insights = []

        if not self.report:
            return insights

        performance = self.report.get(
            "model_performance",
            {}
        )

        model = performance.get(
            "model"
        )

        if model:

            insights.append(
                f"The selected machine learning "
                f"model is {model}."
            )

        metrics = performance.get(
            "metrics",
            {}
        )

        cv_accuracy = metrics.get(
            "accuracy_mean"
        )

        if cv_accuracy is not None:

            insights.append(
                f"Cross-validation accuracy "
                f"was {cv_accuracy * 100:.2f}%."
            )

        cv_f1 = metrics.get(
            "f1_mean"
        )

        if cv_f1 is not None:

            insights.append(
                f"Cross-validation F1 score "
                f"was {cv_f1 * 100:.2f}%."
            )

        return insights

    # ==========================================================
    # FEATURE INSIGHTS
    # ==========================================================

    def feature_insights(self):

        insights = []

        if not self.report:
            return insights

        feature_analysis = (
            self.report.get(
                "feature_analysis",
                {}
            )
        )

        features = (
            feature_analysis.get(
                "top_features",
                []
            )
        )

        if not features:
            return insights

        top_feature = features[0]

        insights.append(
            f"The most influential feature "
            f"in the model is "
            f"{top_feature['feature']} "
            f"with a normalized importance "
            f"of "
            f"{top_feature['importance_percentage']}%."
        )

        if len(features) >= 3:

            top_three = [
                feature["feature"]
                for feature in features[:3]
            ]

            insights.append(
                "The three most influential "
                "features are "
                + ", ".join(top_three)
                + "."
            )

        return insights

    # ==========================================================
    # ERROR INSIGHTS
    # ==========================================================

    def error_insights(self):

        insights = []

        if not self.report:
            return insights

        error_analysis = (
            self.report.get(
                "error_analysis",
                {}
            )
        )

        total = error_analysis.get(
            "total_rows"
        )

        incorrect = error_analysis.get(
            "incorrect_predictions"
        )

        error_percentage = (
            error_analysis.get(
                "error_percentage"
            )
        )

        if (
            total is not None
            and incorrect is not None
        ):

            insights.append(
                f"The model made "
                f"{incorrect} incorrect "
                f"predictions out of "
                f"{total} analyzed rows."
            )

        if error_percentage is not None:

            insights.append(
                f"The diagnostic error rate "
                f"on the analyzed dataset was "
                f"{error_percentage}%."
            )

        return insights

    # ==========================================================
    # SHAP INSIGHTS
    # ==========================================================

    def shap_insights(self):

        insights = []

        if not self.report:
            return insights

        shap_analysis = (
            self.report.get(
                "shap_analysis"
            )
        )

        if not shap_analysis:
            return insights

        positive = (
            shap_analysis.get(
                "positive_factors",
                []
            )
        )

        negative = (
            shap_analysis.get(
                "negative_factors",
                []
            )
        )

        if positive:

            names = [
                item["feature"]
                for item in positive[:3]
            ]

            insights.append(
                "For the analyzed prediction, "
                "the strongest positive "
                "contributors were "
                + ", ".join(names)
                + "."
            )

        if negative:

            names = [
                item["feature"]
                for item in negative[:3]
            ]

            insights.append(
                "The strongest negative "
                "contributors were "
                + ", ".join(names)
                + "."
            )

        return insights

    # ==========================================================
    # WHAT-IF INSIGHTS
    # ==========================================================

    def what_if_insights(self):

        insights = []

        if not self.what_if_result:
            return insights

        result = self.what_if_result

        original = result[
            "original"
        ]

        modified = result[
            "modified"
        ]

        changed_features = result.get(
            "changed_features",
            []
        )

        if changed_features:

            for change in changed_features:

                insights.append(
                    f"The feature "
                    f"{change['feature']} "
                    f"changed from "
                    f"{change['original_value']} "
                    f"to "
                    f"{change['modified_value']}."
                )

        if result.get(
            "prediction_changed",
            False
        ):

            insights.append(
                f"The prediction changed "
                f"from "
                f"{original['prediction']} "
                f"to "
                f"{modified['prediction']} "
                f"after the input modification."
            )

        else:

            insights.append(
                "The prediction remained "
                "unchanged after the input "
                "modification."
            )

        confidence_change = (
            result.get(
                "confidence_change"
            )
        )

        if confidence_change is not None:

            if confidence_change > 0:

                insights.append(
                    f"Model confidence increased "
                    f"by {confidence_change} "
                    f"percentage points."
                )

            elif confidence_change < 0:

                insights.append(
                    f"Model confidence decreased "
                    f"by {abs(confidence_change)} "
                    f"percentage points."
                )

            else:

                insights.append(
                    "Model confidence did not "
                    "change."
                )

        return insights

    # ==========================================================
    # GENERATE ALL INSIGHTS
    # ==========================================================

    def generate(self):

        return {
            "model_insights":
                self.model_insights(),

            "feature_insights":
                self.feature_insights(),

            "error_insights":
                self.error_insights(),

            "shap_insights":
                self.shap_insights(),

            "what_if_insights":
                self.what_if_insights()
        }

    # ==========================================================
    # FLAT INSIGHT LIST
    # ==========================================================

    def all_insights(self):

        grouped = self.generate()

        insights = []

        for category in grouped.values():

            insights.extend(
                category
            )

        return insights