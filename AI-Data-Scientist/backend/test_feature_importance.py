from ml_engine.feature_importance import (
    FeatureImportanceEngine
)


def main():

    print("\n")
    print("=" * 70)
    print(
        "                 AI DATA SCIENTIST"
    )
    print(
        "                  FEATURE IMPORTANCE"
    )
    print("=" * 70)

    engine = FeatureImportanceEngine()

    engine.load_model()

    print(
        "\n✓ Complete ML pipeline loaded"
    )

    result = (
        engine.calculate_importance()
    )

    print("\nMODEL")
    print("-" * 70)

    print(
        f"Model: "
        f"{result['model']}"
    )

    print(
        f"Method: "
        f"{result['method']}"
    )

    print("\nFEATURE IMPORTANCE")
    print("-" * 70)

    for item in result[
        "features"
    ]:

        print(
            f"{item['feature']:<35}"
            f"Importance: "
            f"{item['importance']:<10}"
            f"({item['importance_percentage']}%) "
            f"{item['direction']}"
        )

    print("\nTOP 3 FEATURES")
    print("-" * 70)

    for index, item in enumerate(
        result["features"][:3],
        start=1
    ):

        print(
            f"{index}. "
            f"{item['feature']} "
            f"→ "
            f"{item['importance_percentage']}%"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()