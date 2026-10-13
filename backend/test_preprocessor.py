import pandas as pd

from data_engine.preprocessor import (
    AutomaticPreprocessor
)


def main():

    df = pd.read_csv(
        "datasets/test.csv"
    )

    target_column = "Purchased"

    preprocessor = AutomaticPreprocessor(
        df,
        target_column
    )

    plan = preprocessor.generate_plan()

    print("\n")
    print("=" * 70)
    print("          AUTOMATIC PREPROCESSING ENGINE")
    print("=" * 70)

    print("\nTARGET")
    print("-" * 70)

    print(
        f"Target column: "
        f"{plan['target']}"
    )

    print("\nNUMERICAL FEATURES")
    print("-" * 70)

    if plan["features"]["numerical"]:

        for column in (
            plan["features"]["numerical"]
        ):
            print(f"  • {column}")

    else:

        print("  None")

    print("\nCATEGORICAL FEATURES")
    print("-" * 70)

    if plan["features"]["categorical"]:

        for column in (
            plan["features"]["categorical"]
        ):
            print(f"  • {column}")

    else:

        print("  None")

    print("\nREMOVED COLUMNS")
    print("-" * 70)

    print("ID-like columns:")

    if plan["removed_columns"]["id_like"]:

        for column in (
            plan["removed_columns"]["id_like"]
        ):
            print(f"  ⚠ {column}")

    else:

        print("  ✓ None")

    print("\nConstant columns:")

    if plan["removed_columns"]["constant"]:

        for column in (
            plan["removed_columns"]["constant"]
        ):
            print(f"  ⚠ {column}")

    else:

        print("  ✓ None")

    print("\nMISSING VALUE STRATEGY")
    print("-" * 70)

    numerical_missing = (
        plan["missing_values"]
        ["numerical"]
    )

    categorical_missing = (
        plan["missing_values"]
        ["categorical"]
    )

    if numerical_missing:

        for column, info in (
            numerical_missing.items()
        ):

            print(
                f"  {column}: "
                f"{info['strategy']}"
            )

    if categorical_missing:

        for column, info in (
            categorical_missing.items()
        ):

            print(
                f"  {column}: "
                f"{info['strategy']}"
            )

    if (
        not numerical_missing
        and not categorical_missing
    ):

        print("  ✓ No missing values")

    print("\nTRANSFORMATIONS")
    print("-" * 70)

    print("Numerical:")

    for transformation in (
        plan["transformations"]
        ["numerical"]
    ):

        print(
            f"  • {transformation}"
        )

    print("\nCategorical:")

    for transformation in (
        plan["transformations"]
        ["categorical"]
    ):

        print(
            f"  • {transformation}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()