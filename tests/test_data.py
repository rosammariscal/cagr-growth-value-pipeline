import pandas as pd

from src.data import ROW_ALIASES, pick_row


def frame(labels):
    return pd.DataFrame({"2025-12-31": range(len(labels))}, index=labels)


def test_exact_match_preferred_over_substring():
    df = frame(["Total Operating Income As Reported", "Operating Income"])
    assert pick_row(df, ROW_ALIASES["operating_income"]).name == "Operating Income"


def test_ebit_does_not_match_ebitda():
    df = frame(["EBITDA", "Normalized EBITDA"])
    assert pick_row(df, ("ebit",)) is None


def test_cost_of_revenue_is_not_revenue():
    df = frame(["Cost Of Revenue", "Gross Profit"])
    assert pick_row(df, ROW_ALIASES["revenue"]) is None


def test_whole_word_fallback():
    df = frame(["Free Cash Flow Reported"])
    assert pick_row(df, ROW_ALIASES["fcf"]).name == "Free Cash Flow Reported"


def test_empty_frame():
    assert pick_row(pd.DataFrame(), ("revenue",)) is None
    assert pick_row(None, ("revenue",)) is None
