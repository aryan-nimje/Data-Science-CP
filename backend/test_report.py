import pandas as pd

from data_engine.target_detector import (
    TargetDetector
)

from data_engine.preprocessor import (
    AutomaticPreprocessor
)

from data_engine.preprocessing_pipeline import (
    PreprocessingPipeline
)

from ml_engine.model_trainer import (
    ModelTrainer
)

from ml_engine.error_analyzer import (
    ErrorAnalyzer
)

from ml_engine.feature_importance import (
    FeatureImportanceEngine
)

from ml_engine.explainability import (
    ExplainabilityEngine
)

from ml_engine.what_if import (
    WhatIfEngine
)

from report import (
    ModelInvestigationReport
)


def main():

    print("\n")
    print("=" * 70)
    print(
        "                 AI DATA SCIENTIST"
    )
    print(
        "              MODEL INVESTIGATION"
    )
    print("=" * 70)

    # ======================================================
    # LOAD DATASET
    # ======================================================

    df = pd.read_csv(
        "datasets/test.csv"
    )

    print(
        f"\n✓ Dataset loaded: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    # ======================================================
    # TARGET DETECTION
    # ======================================================

    detector = TargetDetector(df)

    target_result = (
        detector.analyze()
    )

    target = (
        target_result[
            "selected_target"
        ]
    )

    problem_type = (
        target_result[
            "problem"
        ][
            "problem_type"
        ]
    )

    print(
        f"✓ Target detected: {target}"
    )

    print(
        f"✓ Problem type: {problem_type}"
    )

    # ======================================================
    # PREPROCESSING PLAN
    # ======================================================

    planner = AutomaticPreprocessor(
        df,
        target
    )

    plan = (
        planner.generate_plan()
    )

    numerical_features = (
        plan[
            "features"
        ][
            "numerical"
        ]
    )

    categorical_features = (
        plan[
            "features"
        ][
            "categorical"
        ]
    )

    removed_columns = (
        plan[
            "removed_columns"
        ][
            "id_like"
        ]
        +
        plan[
            "removed_columns"
        ][
            "constant"
        ]
    )

    processor = PreprocessingPipeline(
        df,
        target,
        numerical_features,
        categorical_features,
        removed_columns
    )

    preprocessing_pipeline = (
        processor.build()
    )

    X, y = (
        processor.prepare_data()
    )

    # ======================================================
    # MODEL TRAINING
    # ======================================================

    trainer = ModelTrainer(
        preprocessing_pipeline,
        X,
        y,
        problem_type
    )

    training_result = (
        trainer.train()
    )

    trainer.save_model()

    # ======================================================
    # ERROR ANALYSIS
    # ======================================================

    error_analyzer = (
        ErrorAnalyzer()
    )

    error_analyzer.load_model()

    error_result = (
        error_analyzer.analyze(
            df,
            target,
            problem_type
        )
    )

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    importance_engine = (
        FeatureImportanceEngine()
    )

    importance_engine.load_model()

    importance_result = (
        importance_engine
        .calculate_importance()
    )

    # ======================================================
    # SHAP EXPLANATION
    # ======================================================

    explainability_engine = (
        ExplainabilityEngine()
    )

    explainability_engine.load_model()

    sample_input = {

        "Age": 30,

        "Income": 60000,

        "City": "Pune",

        "Experience": 6
    }

    shap_result = (
        explainability_engine
        .explain_prediction(
            sample_input
        )
    )

    # ======================================================
    # WHAT-IF ANALYSIS
    # ======================================================

    what_if_engine = (
        WhatIfEngine()
    )

    what_if_engine.load_model()

    original_data = {

        "Age": 30,

        "Income": 60000,

        "City": "Pune",

        "Experience": 6
    }

    modified_data = {

        "Age": 30,

        "Income": 60000,

        "City": "Pune",

        "Experience": 2
    }

    what_if_result = (
        what_if_engine.compare(
            original_data,
            modified_data
        )
    )

    # ======================================================
    # GENERATE UNIFIED REPORT
    # ======================================================

    report_engine = (
        ModelInvestigationReport(
            training_result,
            error_result,
            importance_result,
            shap_result,
            target,
            problem_type
        )
    )

    # IMPORTANT:
    # Pass What-If result into the report

    report = (
        report_engine
        .generate_report(
            what_if_result
        )
    )

    # IMPORTANT:
    # Save the same complete report

    report_path = (
        report_engine
        .save_json(
            what_if_result=what_if_result
        )
    )

    # ======================================================
    # DISPLAY REPORT
    # ======================================================

    print("\n")
    print("=" * 70)
    print(
        "              AI MODEL INVESTIGATION REPORT"
    )
    print("=" * 70)

    print(
        f"\nTarget: "
        f"{report['target']}"
    )

    print(
        f"Problem Type: "
        f"{report['problem_type']}"
    )

    # ======================================================
    # MODEL
    # ======================================================

    performance = (
        report[
            "model_performance"
        ]
    )

    print("\nMODEL")
    print("-" * 70)

    print(
        f"Selected Model: "
        f"{performance['model']}"
    )

    metrics = (
        performance[
            "metrics"
        ]
    )

    for name, value in (
        metrics.items()
    ):

        print(
            f"{name}: {value}"
        )

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    print("\nTOP FEATURES")
    print("-" * 70)

    for feature in (
        report[
            "feature_analysis"
        ][
            "top_features"
        ]
    ):

        print(
            f"{feature['feature']}: "
            f"{feature['importance_percentage']}%"
        )

    # ======================================================
    # ERROR ANALYSIS
    # ======================================================

    errors = (
        report[
            "error_analysis"
        ]
    )

    print("\nERROR ANALYSIS")
    print("-" * 70)

    print(
        f"Correct Predictions: "
        f"{errors['correct_predictions']}"
    )

    print(
        f"Incorrect Predictions: "
        f"{errors['incorrect_predictions']}"
    )

    print(
        f"Error Percentage: "
        f"{errors['error_percentage']}%"
    )

    # ======================================================
    # SHAP
    # ======================================================

    shap_result_report = (
        report[
            "shap_analysis"
        ]
    )

    if shap_result_report:

        print("\nSHAP EXPLANATION")
        print("-" * 70)

        print(
            "Top Positive Factors:"
        )

        for item in (
            shap_result_report[
                "positive_factors"
            ][:3]
        ):

            print(
                f"  + {item['feature']}: "
                f"{item['shap_value']}"
            )

        print(
            "Top Negative Factors:"
        )

        for item in (
            shap_result_report[
                "negative_factors"
            ][:3]
        ):

            print(
                f"  - {item['feature']}: "
                f"{item['shap_value']}"
            )

    # ======================================================
    # WHAT-IF ANALYSIS
    # ======================================================

    print("\nWHAT-IF ANALYSIS")
    print("-" * 70)

    original = (
        what_if_result[
            "original"
        ]
    )

    modified = (
        what_if_result[
            "modified"
        ]
    )

    print(
        f"Original Prediction: "
        f"{original['prediction']}"
    )

    print(
        f"Original Confidence: "
        f"{original['confidence']}%"
    )

    print(
        f"Modified Prediction: "
        f"{modified['prediction']}"
    )

    print(
        f"Modified Confidence: "
        f"{modified['confidence']}%"
    )

    print("\nChanged Features:")

    for change in (
        what_if_result[
            "changed_features"
        ]
    ):

        print(
            f"  {change['feature']}: "
            f"{change['original_value']} "
            f"→ "
            f"{change['modified_value']}"
        )

    if what_if_result[
        "prediction_changed"
    ]:

        print(
            "\n✓ Prediction changed "
            "after modification."
        )

    else:

        print(
            "\nPrediction remained unchanged."
        )

    confidence_change = (
        what_if_result[
            "confidence_change"
        ]
    )

    if confidence_change is not None:

        if confidence_change < 0:

            print(
                f"Confidence decreased by "
                f"{abs(confidence_change)} "
                f"percentage points."
            )

        elif confidence_change > 0:

            print(
                f"Confidence increased by "
                f"{confidence_change} "
                f"percentage points."
            )

        else:

            print(
                "Confidence remained unchanged."
            )

    # ======================================================
    # AI INSIGHTS
    # ======================================================

    print("\nAI-GENERATED INSIGHTS")
    print("-" * 70)

    ai_insights = (
        report[
            "ai_insights"
        ]
    )

    for category, insights in (
        ai_insights.items()
    ):

        if not insights:
            continue

        print(
            f"\n{category.replace('_', ' ').upper()}"
        )

        for insight in insights:

            print(
                f"  • {insight}"
            )

    # ======================================================
    # REPORT
    # ======================================================

    print("\nREPORT")
    print("-" * 70)

    print(
        f"✓ Saved to: "
        f"{report_path}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()