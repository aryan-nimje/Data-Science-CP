import pandas as pd
import numpy as np


class TargetDetector:

    def __init__(self, dataframe):
        self.df = dataframe

    # ==========================================================
    # CHECK WHETHER COLUMN LOOKS LIKE AN ID
    # ==========================================================

    def is_id_like(self, column):

        series = self.df[column]

        unique_ratio = (
            series.nunique(dropna=True)
            / len(self.df)
        )

        column_name = column.lower()

        id_keywords = [
            "id",
            "index",
            "uuid",
            "identifier",
            "student_no",
            "roll_no",
            "admission_no"
        ]

        if any(
            keyword in column_name
            for keyword in id_keywords
        ):
            return True

        if unique_ratio >= 0.95:
            return True

        return False

    # ==========================================================
    # CHECK CONSTANT COLUMN
    # ==========================================================

    def is_constant(self, column):

        return (
            self.df[column]
            .nunique(dropna=False)
            <= 1
        )

    # ==========================================================
    # TARGET SCORE
    # ==========================================================

    def calculate_target_score(self, column):

        series = self.df[column]

        score = 0
        reasons = []
        rejected = False
        rejection_reason = None

        unique_count = series.nunique(
            dropna=True
        )

        total_rows = len(self.df)

        unique_ratio = (
            unique_count / total_rows
        )

        column_name = column.lower()

        # ------------------------------------------------------
        # 1. Constant column
        # ------------------------------------------------------

        if self.is_constant(column):

            return {
                "column": column,
                "score": 0,
                "status": "Rejected",
                "reason": "Constant column",
                "reasons": [
                    "Column contains only one unique value"
                ]
            }

        # ------------------------------------------------------
        # 2. ID-like column
        # ------------------------------------------------------

        if self.is_id_like(column):

            return {
                "column": column,
                "score": 0,
                "status": "Rejected",
                "reason": "ID-like column",
                "reasons": [
                    "Column has extremely high uniqueness "
                    "or appears to be an identifier"
                ]
            }

        # ------------------------------------------------------
        # 3. Target keywords
        # ------------------------------------------------------

        target_keywords = [
            "target",
            "label",
            "class",
            "output",
            "result",
            "prediction",
            "purchased",
            "churn",
            "price",
            "salary",
            "sales",
            "score"
        ]

        for keyword in target_keywords:

            if keyword in column_name:

                score += 35

                reasons.append(
                    f"Column name contains "
                    f"target keyword '{keyword}'"
                )

                break

        # ------------------------------------------------------
        # 4. Last column
        # ------------------------------------------------------

        if column == self.df.columns[-1]:

            score += 10

            reasons.append(
                "Column appears as the final "
                "dataset column"
            )

        # ------------------------------------------------------
        # 5. Categorical target
        # ------------------------------------------------------

        if (
            series.dtype == "object"
            or str(series.dtype) == "category"
        ):

            if 2 <= unique_count <= 20:

                score += 25

                reasons.append(
                    "Categorical column has a "
                    "reasonable number of classes"
                )

            elif unique_ratio < 0.1:

                score += 10

                reasons.append(
                    "Categorical column has "
                    "relatively low cardinality"
                )

        # ------------------------------------------------------
        # 6. Numeric target
        # ------------------------------------------------------

        elif pd.api.types.is_numeric_dtype(series):

            # Numeric regression target
            if unique_count > 20:

                score += 20

                reasons.append(
                    "Numeric column has sufficient "
                    "unique values for regression"
                )

            # Small numeric cardinality is NOT strongly
            # considered a target.
            elif unique_count <= 20:

                score += 5

                reasons.append(
                    "Numeric column has limited "
                    "unique values"
                )

        # ------------------------------------------------------
        # 7. Avoid extremely high-cardinality columns
        # ------------------------------------------------------

        if unique_ratio > 0.90:

            score -= 20

            reasons.append(
                "Very high uniqueness reduces "
                "target likelihood"
            )

        # ------------------------------------------------------
        # Final score
        # ------------------------------------------------------

        score = max(
            0,
            min(score, 100)
        )

        if score >= 60:

            status = "Recommended"

        elif score >= 30:

            status = "Possible"

        else:

            status = "Low Probability"

        return {
            "column": column,
            "score": score,
            "status": status,
            "reason": rejection_reason,
            "reasons": reasons
        }

    # ==========================================================
    # FIND TARGET CANDIDATES
    # ==========================================================

    def detect_candidates(self):

        candidates = []

        for column in self.df.columns:

            result = self.calculate_target_score(
                column
            )

            candidates.append(result)

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return candidates

    # ==========================================================
    # DETECT CLASSIFICATION / REGRESSION
    # ==========================================================

    def detect_problem_type(
        self,
        target_column
    ):

        if target_column not in self.df.columns:

            return {
                "error": "Target column not found"
            }

        target = self.df[target_column]

        unique_count = target.nunique(
            dropna=True
        )

        # Categorical
        if (
            target.dtype == "object"
            or str(target.dtype) == "category"
        ):

            if unique_count == 2:

                problem_type = (
                    "Binary Classification"
                )

            else:

                problem_type = (
                    "Multiclass Classification"
                )

        # Numeric
        elif pd.api.types.is_numeric_dtype(
            target
        ):

            # Small number of unique values
            # can represent classification.
            if unique_count <= 10:

                if unique_count == 2:

                    problem_type = (
                        "Binary Classification"
                    )

                else:

                    problem_type = (
                        "Multiclass Classification"
                    )

            else:

                problem_type = "Regression"

        else:

            problem_type = "Unknown"

        return {
            "target": target_column,
            "problem_type": problem_type,
            "unique_values": int(unique_count)
        }

    # ==========================================================
    # CLASS BALANCE
    # ==========================================================

    def analyze_class_balance(
        self,
        target_column
    ):

        if target_column not in self.df.columns:

            return {
                "error": "Target column not found"
            }

        target = self.df[target_column]

        distribution = (
            target.value_counts(
                normalize=True,
                dropna=False
            ) * 100
        ).round(2)

        imbalance = False

        if len(distribution) >= 2:

            largest = distribution.max()
            smallest = distribution.min()

            if largest >= 70:
                imbalance = True

            if smallest <= 5:
                imbalance = True

        return {
            "distribution":
                distribution.to_dict(),

            "imbalanced":
                imbalance
        }

    # ==========================================================
    # COMPLETE ANALYSIS
    # ==========================================================

    def analyze(self):

        candidates = self.detect_candidates()

        # Find first non-rejected candidate
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate["status"] != "Rejected"
        ]

        if not valid_candidates:

            return {
                "candidates": candidates,
                "selected_target": None,
                "confidence": 0,
                "problem": {
                    "error":
                    "No suitable target detected"
                },
                "class_balance": None
            }

        best_candidate = valid_candidates[0]

        target_column = (
            best_candidate["column"]
        )

        problem = self.detect_problem_type(
            target_column
        )

        balance = None

        if "Classification" in problem.get(
            "problem_type",
            ""
        ):

            balance = (
                self.analyze_class_balance(
                    target_column
                )
            )

        return {
            "candidates": candidates,
            "selected_target": target_column,
            "confidence":
                best_candidate["score"],
            "problem": problem,
            "class_balance": balance
        }