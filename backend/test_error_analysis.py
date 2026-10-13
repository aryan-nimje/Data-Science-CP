import pandas as pd

from ml_engine.error_analyzer import (
    ErrorAnalyzer
)


def main():

    print("\n")
    print("=" * 70)
    print(
        "                 AI DATA SCIENTIST"
    )
    print(
        "                    ERROR ANALYSIS"
    )
    print("=" * 70)

    # ==================================================
    # LOAD DATA
    # ==================================================

    df = pd.read_csv(
        "datasets/test.csv"
    )

    target_column = "Purchased"

    problem_type = (
        "Binary Classification"
    )

    print(
        f"\nDataset rows: "
        f"{len(df)}"
    )

    print(
        f"Target: "
        f"{target_column}"
    )

    print(
        f"Problem: "
        f"{problem_type}"
    )

    # ==================================================
    # CREATE ANALYZER
    # ==================================================

    analyzer = ErrorAnalyzer()

    analyzer.load_model()

    print(
        "\n✓ Complete ML pipeline loaded"
    )

    # ==================================================
    # ANALYZE
    # ==================================================

    result = analyzer.analyze(
        df,
        target_column,
        problem_type
    )

    # ==================================================
    # PERFORMANCE
    # ==================================================

    print("\nMODEL PERFORMANCE")
    print("-" * 70)

    metrics = result[
        "metrics"
    ]

    for name, value in metrics.items():

        print(
            f"{name}: {value}"
        )

    # ==================================================
    # PREDICTION COUNTS
    # ==================================================

    print("\nPREDICTION ANALYSIS")
    print("-" * 70)

    print(
        f"Total rows: "
        f"{result['total_rows']}"
    )

    print(
        f"Correct predictions: "
        f"{result['correct_predictions']}"
    )

    print(
        f"Incorrect predictions: "
        f"{result['incorrect_predictions']}"
    )

    print(
        f"Error percentage: "
        f"{result['error_percentage']}%"
    )

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    print("\nCONFUSION MATRIX")
    print("-" * 70)

    labels = result[
        "confusion_matrix"
    ]["labels"]

    matrix = result[
        "confusion_matrix"
    ]["matrix"]

    print(
        f"Labels: {labels}"
    )

    for row in matrix:

        print(row)

    # ==================================================
    # CLASS METRICS
    # ==================================================

    print("\nCLASS-WISE PERFORMANCE")
    print("-" * 70)

    for label, values in (
        result[
            "class_metrics"
        ].items()
    ):

        print(
            f"\nClass: {label}"
        )

        print(
            f"  Precision: "
            f"{values['precision']}"
        )

        print(
            f"  Recall: "
            f"{values['recall']}"
        )

        print(
            f"  F1 Score: "
            f"{values['f1_score']}"
        )

        print(
            f"  Support: "
            f"{values['support']}"
        )

    # ==================================================
    # MISCLASSIFIED ROWS
    # ==================================================

    print("\nMISCLASSIFIED ROWS")
    print("-" * 70)

    errors = result[
        "misclassified_rows"
    ]

    if errors:

        for row in errors:

            print(
                f"Row {row['Row']}: "
                f"Actual={row['Actual']} "
                f"| Predicted={row['Predicted']}"
            )

    else:

        print(
            "✓ No misclassified rows"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()