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


def main():

    # ------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------

    df = pd.read_csv(
        "datasets/test.csv"
    )

    # ------------------------------------------------------
    # Detect target
    # ------------------------------------------------------

    detector = TargetDetector(df)

    target_result = detector.analyze()

    target = (
        target_result["selected_target"]
    )

    # ------------------------------------------------------
    # Generate preprocessing plan
    # ------------------------------------------------------

    planner = AutomaticPreprocessor(
        df,
        target
    )

    plan = planner.generate_plan()

    numerical_features = (
        plan["features"]["numerical"]
    )

    categorical_features = (
        plan["features"]["categorical"]
    )

    removed_columns = (
        plan["removed_columns"]
        ["id_like"]
        +
        plan["removed_columns"]
        ["constant"]
    )

    # ------------------------------------------------------
    # Build actual pipeline
    # ------------------------------------------------------

    processor = PreprocessingPipeline(

        df,

        target,

        numerical_features,

        categorical_features,

        removed_columns
    )

    # ------------------------------------------------------
    # Transform data
    # ------------------------------------------------------

    X_transformed, y = (
        processor.fit_transform()
    )

    feature_names = (
        processor.get_feature_names()
    )

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("          ACTUAL PREPROCESSING PIPELINE")
    print("=" * 70)

    print("\nTARGET")
    print("-" * 70)

    print(
        f"Target: {target}"
    )

    print(
        f"Target rows: {len(y)}"
    )

    print("\nORIGINAL DATA")
    print("-" * 70)

    print(
        f"Rows: {df.shape[0]}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    print("\nTRANSFORMED DATA")
    print("-" * 70)

    print(
        f"Rows: {X_transformed.shape[0]}"
    )

    print(
        f"Columns: {X_transformed.shape[1]}"
    )

    print("\nGENERATED FEATURES")
    print("-" * 70)

    for feature in feature_names:

        print(
            f"  • {feature}"
        )

    print("\nPIPELINE STATUS")
    print("-" * 70)

    print(
        "✓ Missing values handled"
    )

    print(
        "✓ Numerical features scaled"
    )

    print(
        "✓ Categorical features encoded"
    )

    print(
        "✓ Constant columns removed"
    )

    print(
        "✓ Feature transformation completed"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()