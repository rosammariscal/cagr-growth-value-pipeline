import numpy as np
import pandas as pd
import pytest

from src.scoring import apply_gates, composite, rank_01, score

GATES = {
    "min_usable_growth_metrics": 2,
    "min_operating_margin": 0.08,
    "max_debt_to_equity": 2.5,
    "min_market_cap": 1e10,
}


def universe(**overrides):
    base = {
        "symbol": ["A", "B", "C", "D"],
        "sector": ["Technology"] * 4,
        "industry": ["Software"] * 4,
        "revenue_cagr": [0.30, 0.20, 0.10, 0.05],
        "earnings_cagr": [0.30, 0.20, 0.10, 0.05],
        "fcf_cagr": [0.30, 0.20, 0.10, 0.05],
        "fcf_yield": [0.01, 0.02, 0.03, 0.04],
        "trailing_pe": [40.0, 30.0, 20.0, 10.0],
        "price_to_book": [10.0, 8.0, 6.0, 4.0],
        "operating_margin": [0.30, 0.25, 0.20, 0.15],
        "debt_to_equity": [0.1, 0.2, 0.3, 0.4],
        "negative_equity": [False] * 4,
        "usable_growth_metrics": [3, 3, 3, 3],
        "market_cap": [1e11] * 4,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_rank_01_keeps_missing_as_missing():
    r = rank_01(pd.Series([1.0, np.nan, 3.0]), higher_is_better=False)
    assert np.isnan(r.iloc[1])
    assert r.iloc[0] > r.iloc[2]


def test_composite_reweights_over_available_components():
    ranks = pd.DataFrame({"growth_rank": [1.0, 1.0], "value_rank": [np.nan, 0.0], "quality_rank": [0.5, 0.5]})
    s = composite(ranks, {"growth": 0.45, "value": 0.35, "quality": 0.20})
    # Row 0 has no value metrics: (0.45 * 1 + 0.20 * 0.5) / 0.65
    assert s.iloc[0] == pytest.approx((0.45 + 0.10) / 0.65)
    # Row 1 is penalised by a genuinely low value rank, not by a missing one.
    assert s.iloc[1] == pytest.approx(0.45 + 0.10)


def test_composite_all_missing_is_nan():
    ranks = pd.DataFrame({"growth_rank": [np.nan], "value_rank": [np.nan], "quality_rank": [np.nan]})
    assert np.isnan(composite(ranks, {}).iloc[0])


def test_missing_value_metric_does_not_zero_the_score():
    df = universe(fcf_yield=[0.01, np.nan, 0.03, 0.04], trailing_pe=[40.0, np.nan, 20.0, 10.0],
                  price_to_book=[10.0, np.nan, 6.0, 4.0])
    out = score(df, {}, GATES).set_index("symbol")
    assert np.isnan(out.loc["B", "value_rank"])
    assert out.loc["B", "score"] > 0.5


def test_gates_and_reasons():
    df = universe(
        operating_margin=[0.30, 0.02, np.nan, 0.15],
        negative_equity=[False, False, False, True],
        debt_to_equity=[0.1, 0.2, 0.3, np.nan],
    )
    g = apply_gates(df, GATES).set_index(df["symbol"])
    assert g.loc["A", "passes_gates"]
    assert "low_or_missing_margin" in g.loc["B", "gate_failures"]
    assert "low_or_missing_margin" in g.loc["C", "gate_failures"]
    assert "negative_equity" in g.loc["D", "gate_failures"]


def test_industry_and_sector_exclusions():
    df = universe(sector=["Technology", "Financial Services", "Financial Services", "Technology"],
                  industry=["Software", "Banks - Diversified", "Credit Services", "Software"])
    g = apply_gates(df, GATES, {"industry_keywords": ["Banks", "Insurance"]}).set_index(df["symbol"])
    assert g["passes_gates"].tolist() == [True, False, True, True]
    g2 = apply_gates(df, GATES, {"sectors": ["Financial Services"]}).set_index(df["symbol"])
    assert g2["passes_gates"].tolist() == [True, False, False, True]


def test_ranks_are_computed_among_eligible_names_only():
    df = universe(market_cap=[1e11, 1e11, 1e11, 1e9])  # D fails market-cap gate
    out = score(df, {}, GATES).set_index("symbol")
    assert np.isnan(out.loc["D", "score"])
    assert np.isnan(out.loc["D", "growth_rank"])
    assert out.loc["A", "growth_rank"] == pytest.approx(1.0)
    assert out.loc["C", "growth_rank"] == pytest.approx(1 / 3)


def test_score_is_sorted_and_bounded():
    out = score(universe(), {}, GATES)
    s = out["score"].dropna()
    assert s.is_monotonic_decreasing
    assert ((s > 0) & (s <= 1)).all()
