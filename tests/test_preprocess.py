import pandas as pd
import pytest

from src.preprocess import clean_narrative, prepare_complaints


def test_clean_narrative_removes_boilerplate_and_normalizes_text():
    result = clean_narrative("I am writing to file a complaint... Charged @ twice! https://x.test")
    assert result == "... charged twice!"


def test_prepare_filters_and_maps_historical_products():
    raw = pd.DataFrame(
        {
            "Complaint ID": [1, 2, 3, 4],
            "Product": ["Credit card or prepaid card", "Checking or savings account", "Mortgage", "Credit card"],
            "Consumer complaint narrative": ["Charged twice", "Cannot withdraw", "Escrow issue", None],
        }
    )
    result = prepare_complaints(raw)
    assert result["product_category"].tolist() == ["Credit Card", "Savings Account"]
    assert result["narrative_word_count"].tolist() == [2, 2]


def test_prepare_reports_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_complaints(pd.DataFrame({"Product": []}))

