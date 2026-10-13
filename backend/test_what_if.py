from ml_engine.what_if import (
    WhatIfEngine
)


def main():

    print("\n")
    print("=" * 70)
    print(
        "                 AI DATA SCIENTIST"
    )
    print(
        "                    WHAT-IF ANALYSIS"
    )
    print("=" * 70)

    engine = WhatIfEngine()

    engine.load_model()

    print(
        "\n✓ Complete ML pipeline loaded"
    )

    # ======================================================
    # ORIGINAL SCENARIO
    # ======================================================

    original_data = {

        "Age": 30,

        "Income": 60000,

        "City": "Pune",

        "Experience": 6
    }

    # ======================================================
    # MODIFIED SCENARIO
    # ======================================================

    modified_data = {

        "Age": 30,

        "Income": 60000,

        "City": "Pune",

        "Experience": 2
    }

    print("\nORIGINAL SCENARIO")
    print("-" * 70)

    for feature, value in (
        original_data.items()
    ):

        print(
            f"{feature}: {value}"
        )

    print("\nMODIFIED SCENARIO")
    print("-" * 70)

    for feature, value in (
        modified_data.items()
    ):

        print(
            f"{feature}: {value}"
        )

    # ======================================================
    # WHAT-IF COMPARISON
    # ======================================================

    result = engine.compare(
        original_data,
        modified_data
    )

    # ======================================================
    # RESULTS
    # ======================================================

    print("\nORIGINAL PREDICTION")
    print("-" * 70)

    print(
        f"Prediction: "
        f"{result['original']['prediction']}"
    )

    if result[
        "original"
    ][
        "confidence"
    ] is not None:

        print(
            f"Confidence: "
            f"{result['original']['confidence']}%"
        )

    print("\nMODIFIED PREDICTION")
    print("-" * 70)

    print(
        f"Prediction: "
        f"{result['modified']['prediction']}"
    )

    if result[
        "modified"
    ][
        "confidence"
    ] is not None:

        print(
            f"Confidence: "
            f"{result['modified']['confidence']}%"
        )

    # ======================================================
    # CHANGED FEATURES
    # ======================================================

    print("\nCHANGED FEATURES")
    print("-" * 70)

    for change in (
        result[
            "changed_features"
        ]
    ):

        print(
            f"{change['feature']}: "
            f"{change['original_value']} "
            f"→ "
            f"{change['modified_value']}"
        )

    # ======================================================
    # PREDICTION CHANGE
    # ======================================================

    print("\nWHAT-IF RESULT")
    print("-" * 70)

    if result[
        "prediction_changed"
    ]:

        print(
            "✓ Prediction changed "
            "after modifying the input."
        )

    else:

        print(
            "Prediction remained unchanged."
        )

    confidence_change = (
        result[
            "confidence_change"
        ]
    )

    if confidence_change is not None:

        if confidence_change > 0:

            print(
                f"Confidence increased by "
                f"{confidence_change}%"
            )

        elif confidence_change < 0:

            print(
                f"Confidence decreased by "
                f"{abs(confidence_change)}%"
            )

        else:

            print(
                "Confidence remained unchanged."
            )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()