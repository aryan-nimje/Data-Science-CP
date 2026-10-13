import pandas as pd

from data_engine.target_detector import TargetDetector


def main():

    df = pd.read_csv(
        "datasets/test.csv"
    )

    detector = TargetDetector(df)

    result = detector.analyze()

    print("\n")
    print("=" * 70)
    print("             TARGET DETECTION ENGINE")
    print("=" * 70)

    print("\nTARGET CANDIDATES")
    print("-" * 70)

    for candidate in result["candidates"]:

        print(
            f"\n{candidate['column']}"
        )

        print(
            f"Confidence: "
            f"{candidate['score']}%"
        )

        for reason in candidate["reasons"]:
            print(
                f"  • {reason}"
            )

    print("\n")
    print("=" * 70)
    print("AUTOMATIC PROBLEM DETECTION")
    print("=" * 70)

    print(
        f"\nSelected Target: "
        f"{result['selected_target']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']}%"
    )

    problem = result["problem"]

    print(
        f"Problem Type: "
        f"{problem['problem_type']}"
    )

    print(
        f"Unique Target Values: "
        f"{problem['unique_values']}"
    )

    if result["class_balance"]:

        print("\nCLASS DISTRIBUTION")

        for value, percentage in (
            result["class_balance"]
            ["distribution"]
            .items()
        ):

            print(
                f"  {value}: "
                f"{percentage}%"
            )

        print(
            "\nClass Imbalance: "
            f"{result['class_balance']['imbalanced']}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()