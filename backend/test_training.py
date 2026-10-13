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


def main():

    # ==================================================
    # LOAD DATASET
    # ==================================================

    df = pd.read_csv(
        "datasets/test.csv"
    )

    # ==================================================
    # TARGET DETECTION
    # ==================================================

    detector = TargetDetector(df)

    target_result = detector.analyze()

    target = (
        target_result["selected_target"]
    )

    problem_type = (
        target_result["problem"]
        ["problem_type"]
    )

    # ==================================================
    # AUTOMATIC PREPROCESSING PLAN
    # ==================================================

    planner = AutomaticPreprocessor(
        df,
        target
    )

    plan = planner.generate_plan()

    numerical_features = (
        plan["features"]
        ["numerical"]
    )

    categorical_features = (
        plan["features"]
        ["categorical"]
    )

    removed_columns = (
        plan["removed_columns"]
        ["id_like"]
        +
        plan["removed_columns"]
        ["constant"]
    )

    # ==================================================
    # CREATE PREPROCESSING PIPELINE
    # ==================================================

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

    # ==================================================
    # PREPARE RAW DATA
    # ==================================================

    X, y = (
        processor.prepare_data()
    )

    # ==================================================
    # AUTOMATED MODEL TRAINING
    # ==================================================

    trainer = ModelTrainer(
        preprocessing_pipeline,
        X,
        y,
        problem_type
    )

    results = trainer.train()

    # ==================================================
    # SAVE COMPLETE PIPELINE
    # ==================================================

    trainer.save_model()

    # ==================================================
    # FINAL RESULT
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "                 AI DATA SCIENTIST"
    )
    print(
        "                   FINAL RESULT"
    )
    print("=" * 70)

    print(
        f"\nTarget: {target}"
    )

    print(
        f"Problem: {problem_type}"
    )

    print(
        f"Best Model: "
        f"{results['best_model']}"
    )

    print(
        "\nData Leakage Protection: ENABLED"
    )

    print(
        "\nComplete Pipeline:"
    )

    print(
        "Raw Data → Preprocessing → "
        "Best Model → Prediction"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()