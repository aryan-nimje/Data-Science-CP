import json

from ml_engine.insight_engine import (
    InsightEngine
)


def main():

    print("\n")
    print("=" * 70)
    print(
        "                 AI DATA SCIENTIST"
    )
    print(
        "                  INSIGHT ENGINE"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # LOAD EXISTING REPORT
    # ------------------------------------------------------

    with open(
        "reports/model_investigation_report.json",
        "r",
        encoding="utf-8"
    ) as file:

        report = json.load(
            file
        )

    # ------------------------------------------------------
    # WHAT-IF RESULT
    # ------------------------------------------------------

    what_if_result = {

        "original": {
            "prediction": "Yes",
            "confidence": 73.77
        },

        "modified": {
            "prediction": "No",
            "confidence": 52.34
        },

        "changed_features": [
            {
                "feature": "Experience",
                "original_value": 6,
                "modified_value": 2
            }
        ],

        "prediction_changed": True,

        "confidence_change": -21.43
    }

    # ------------------------------------------------------
    # GENERATE INSIGHTS
    # ------------------------------------------------------

    engine = InsightEngine(
        report,
        what_if_result
    )

    result = engine.generate()

    # ------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------

    print("\nMODEL INSIGHTS")
    print("-" * 70)

    for insight in (
        result[
            "model_insights"
        ]
    ):

        print(
            f"• {insight}"
        )

    print("\nFEATURE INSIGHTS")
    print("-" * 70)

    for insight in (
        result[
            "feature_insights"
        ]
    ):

        print(
            f"• {insight}"
        )

    print("\nERROR INSIGHTS")
    print("-" * 70)

    for insight in (
        result[
            "error_insights"
        ]
    ):

        print(
            f"• {insight}"
        )

    print("\nSHAP INSIGHTS")
    print("-" * 70)

    for insight in (
        result[
            "shap_insights"
        ]
    ):

        print(
            f"• {insight}"
        )

    print("\nWHAT-IF INSIGHTS")
    print("-" * 70)

    for insight in (
        result[
            "what_if_insights"
        ]
    ):

        print(
            f"• {insight}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()