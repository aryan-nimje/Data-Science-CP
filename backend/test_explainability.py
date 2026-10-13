from ml_engine.explainability import (
    ExplainabilityEngine
)


def main():

    print("\n")
    print("=" * 70)
    print(
        "                 AI DATA SCIENTIST"
    )
    print(
        "                  EXPLAINABLE AI"
    )
    print("=" * 70)

    # ==================================================
    # LOAD ENGINE
    # ==================================================

    engine = ExplainabilityEngine()

    engine.load_model()

    print(
        "\n✓ Complete ML pipeline loaded"
    )

    # ==================================================
    # DATA TO EXPLAIN
    # ==================================================

    customer = {
        "Age": 30,
        "Income": 60000,
        "City": "Pune",
        "Experience": 6
    }

    print("\nDATA BEING EXPLAINED")
    print("-" * 70)

    for feature, value in customer.items():

        print(
            f"{feature}: {value}"
        )

    # ==================================================
    # EXPLAIN
    # ==================================================

    explanation = (
        engine.explain_prediction(
            customer
        )
    )

    # ==================================================
    # MODEL INFORMATION
    # ==================================================

    print("\nMODEL")
    print("-" * 70)

    print(
        f"Model: "
        f"{explanation['model']}"
    )

    print(
        f"SHAP Base Value: "
        f"{explanation['base_value']}"
    )

    # ==================================================
    # FEATURE CONTRIBUTIONS
    # ==================================================

    print("\nFEATURE CONTRIBUTIONS")
    print("-" * 70)

    for item in explanation[
        "features"
    ]:

        value = item[
            "shap_value"
        ]

        if value > 0:
            direction = "POSITIVE"
        elif value < 0:
            direction = "NEGATIVE"
        else:
            direction = "NEUTRAL"

        print(
            f"{item['feature']:<35} "
            f"{value:>10.6f} "
            f"({direction})"
        )

    # ==================================================
    # POSITIVE FACTORS
    # ==================================================

    print("\nTOP POSITIVE FACTORS")
    print("-" * 70)

    if explanation[
        "positive_factors"
    ]:

        for item in explanation[
            "positive_factors"
        ][:5]:

            print(
                f"+ {item['feature']}: "
                f"{item['shap_value']:.6f}"
            )

    else:

        print(
            "No positive factors detected."
        )

    # ==================================================
    # NEGATIVE FACTORS
    # ==================================================

    print("\nTOP NEGATIVE FACTORS")
    print("-" * 70)

    if explanation[
        "negative_factors"
    ]:

        for item in explanation[
            "negative_factors"
        ][:5]:

            print(
                f"- {item['feature']}: "
                f"{item['shap_value']:.6f}"
            )

    else:

        print(
            "No negative factors detected."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()