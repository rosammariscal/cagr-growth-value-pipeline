import numpy as np
import pandas as pd
import pytest

from src.features import cagr, extract_features, series_cagr

YEAR_ENDS = pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"])


def annual(values):
    return pd.Series(values, index=YEAR_ENDS[-len(values):], dtype=float)


def test_cagr_basic():
    assert cagr(100, 200, 4) == pytest.approx(2 ** 0.25 - 1)


@pytest.mark.parametrize(
    "start,end,years",
    [(-100, -50, 4), (-50, -100, 4), (-10, 12, 4), (10, -12, 4), (0, 10, 4), (10, 0, 4), (10, 20, 0), (np.nan, 1, 2)],
)
def test_cagr_rejects_non_positive_or_invalid(start, end, years):
    assert cagr(start, end, years) is None


def test_series_cagr_uses_dates():
    res = series_cagr(annual([100, 110, 130, 150, 200]))
    assert res["years"] == pytest.approx(4.0, abs=0.01)
    assert res["cagr"] == pytest.approx(2 ** (1 / 4.0014) - 1, rel=1e-3)
    assert res["reason"] is None


def test_series_cagr_gap_keeps_true_horizon():
    res = series_cagr(annual([100, np.nan, np.nan, np.nan, 200]))
    assert res["n_obs"] == 2
    assert res["years"] == pytest.approx(4.0, abs=0.01)
    assert res["cagr"] == pytest.approx(0.189, abs=1e-3)  # not 100%


def test_series_cagr_sign_change():
    assert series_cagr(annual([-10, 5, 8, 9, 12]))["reason"] == "sign_change"
    assert series_cagr(annual([10, 5, 8, 9, -12]))["reason"] == "sign_change"


def test_series_cagr_both_negative_is_rejected():
    for values in ([-100, -80, -70, -60, -50], [-50, -60, -70, -80, -100]):
        res = series_cagr(annual(values))
        assert res["cagr"] is None
        assert res["reason"] == "non_positive_endpoint"


def test_series_cagr_zero_endpoint():
    assert series_cagr(annual([0, 5, 8, 9, 12]))["reason"] == "non_positive_endpoint"


def test_series_cagr_short_and_missing():
    assert series_cagr(annual([np.nan, np.nan, np.nan, np.nan, 12]))["reason"] == "short_series"
    assert series_cagr(None)["reason"] == "missing_row"
    assert series_cagr(pd.Series(dtype=float))["reason"] == "missing_row"


def test_series_cagr_respects_max_years():
    res = series_cagr(annual([1, 100, 110, 121, 133.1]), max_years=3)
    assert res["years"] == pytest.approx(3.0, abs=0.01)
    assert res["cagr"] == pytest.approx(0.10, abs=1e-3)


def test_series_cagr_non_datetime_index_falls_back_to_count():
    res = series_cagr(pd.Series([100.0, 121.0], index=["FY1", "FY2"]))
    assert res["years"] == 1.0
    assert res["cagr"] == pytest.approx(0.21)


def _bundle(equity=50.0, fcf=10.0, pe=20.0, mcap=1000.0, ccy="USD", fin_ccy="USD"):
    cols = YEAR_ENDS[-4:]
    income = pd.DataFrame(
        {c: v for c, v in zip(cols, [[100, 10, 15], [110, 11, 16], [120, 12, 18], [130, 13, 20]], strict=True)},
        index=["Total Revenue", "Net Income", "Operating Income"],
    )
    cash = pd.DataFrame({c: [fcf] for c in cols}, index=["Free Cash Flow"])
    balance = pd.DataFrame({c: [20.0, equity] for c in cols}, index=["Total Debt", "Stockholders Equity"])
    info = {"marketCap": mcap, "trailingPE": pe, "currency": ccy, "financialCurrency": fin_ccy}
    return {"symbol": "TEST", "info": info, "income": income, "cash": cash, "balance": balance}


def test_extract_features_happy_path():
    f = extract_features(_bundle())
    assert f["price_to_book"] == pytest.approx(1000 / 50)
    assert f["debt_to_equity"] == pytest.approx(20 / 50)
    assert f["fcf_yield"] == pytest.approx(10 / 1000)
    assert f["operating_margin"] == pytest.approx(20 / 130)
    assert f["usable_growth_metrics"] == 3  # constant FCF -> 0% CAGR is valid
    assert f["fcf_cagr"] == pytest.approx(0.0)


def test_negative_equity_is_flagged_not_ranked():
    f = extract_features(_bundle(equity=-5.0))
    assert f["negative_equity"] is True
    assert f["debt_to_equity"] is None and f["price_to_book"] is None
    assert "negative_equity" in f["data_flags"]


def test_non_positive_pe_and_fcf_dropped():
    f = extract_features(_bundle(pe=-7.0, fcf=-3.0))
    assert f["trailing_pe"] is None and f["fcf_yield"] is None
    assert "pe_non_positive" in f["data_flags"] and "fcf_non_positive" in f["data_flags"]


def test_sanity_bounds_drop_absurd_pb():
    # Yahoo reported BRK-B P/B as ~0.001 (book value per A share vs B-share
    # price). We compute P/B ourselves and still guard with sanity bounds.
    f = extract_features(_bundle(equity=1e7))  # P/B = 1e-4
    assert f["price_to_book"] is None
    assert "price_to_book_out_of_bounds" in f["data_flags"]


def test_currency_mismatch_blocks_market_ratios():
    f = extract_features(_bundle(ccy="USD", fin_ccy="TWD"))
    assert f["price_to_book"] is None and f["fcf_yield"] is None
    assert "currency_mismatch" in f["data_flags"]
