from ml_engine.prediction_engine import (
    PredictionEngine
)


def main():

    print("\n")
    print("=" * 70)
    print(
        "                 AI DATA SCIENTIST"
    )
    print(
        "                    PREDICTION ENGINE"
    )
    print("=" * 70)

    # ==================================================
    # LOAD ENGINE
    # ==================================================

    engine = PredictionEngine()

    engine.load_model()

    print(
        "\n✓ Complete ML pipeline loaded"
    )

    # ==================================================
    # NEW RAW DATA
    # ==================================================

    new_customer = {
        "Age": 30,
        "Income": 60000,
        "City": "Pune",
        "Experience": 6
    }

    print("\nNEW DATA")
    print("-" * 70)

    for feature, value in (
        new_customer.items()
    ):
        print(
            f"{feature}: {value}"
        )

    # ==================================================
    # PREDICT
    # ==================================================

    result = (
        engine.predict_single(
            new_customer
        )
    )

    print("\nPREDICTION")
    print("-" * 70)

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    if "confidence" in result:

        print(
            f"Confidence: "
            f"{result['confidence']}%"
        )

    if "class_probabilities" in result:

        print(
            "\nClass Probabilities:"
        )

        for (
            class_name,
            probability
        ) in result[
            "class_probabilities"
        ].items():

            print(
                f"  {class_name}: "
                f"{probability}%"
            )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()