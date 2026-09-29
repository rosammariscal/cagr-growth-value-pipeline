import numpy as np
import pandas as pd
import pytest

from src.portfolio import allocate, water_fill


def scored(n):
    return pd.DataFrame({"symbol": [f"T{i}" for i in range(n)], "score": np.linspace(1.0, 0.1, n)})


@pytest.mark.parametrize("n,cap", [(8, 0.18), (8, 0.13), (10, 0.12), (4, 0.30), (20, 0.06)])
def test_water_fill_sums_to_one_and_respects_cap(n, cap):
    raw = pd.Series(np.linspace(1.0, 0.1, n) ** 2)
    w = water_fill(raw, cap)
    assert w.sum() == pytest.approx(1.0)
    assert (w <= cap + 1e-9).all()
    assert (w >= 0).all()


def test_water_fill_preserves_order_and_proportions_of_free_names():
    raw = pd.Series([10.0, 1.0, 1.0, 1.0, 1.0])
    w = water_fill(raw, 0.4)
    assert w.iloc[0] == pytest.approx(0.4)
    assert w.iloc[1:].tolist() == pytest.approx([0.15] * 4)


def test_water_fill_no_cap_binding_is_plain_normalisation():
    raw = pd.Series([1.0, 2.0, 3.0])
    assert water_fill(raw, 0.9).tolist() == pytest.approx([1 / 6, 2 / 6, 3 / 6])


def test_infeasible_cap_relaxes_to_equal_weight_with_warning():
    with pytest.warns(RuntimeWarning, match="infeasible"):
        w = water_fill(pd.Series([3.0, 2.0, 1.0, 1.0, 1.0]), 0.18)
    assert w.sum() == pytest.approx(1.0)
    assert w.tolist() == pytest.approx([0.2] * 5)


@pytest.mark.parametrize("method", ["equal", "score"])
def test_allocate_top_n(method):
    h = allocate(scored(15), top_n=8, max_weight=0.18, method=method)
    assert len(h) == 8
    assert h["weight"].sum() == pytest.approx(1.0)
    assert h["weight"].max() <= 0.18 + 1e-9
    assert h["symbol"].tolist() == [f"T{i}" for i in range(8)]


def test_allocate_skips_unscored_and_handles_empty():
    df = scored(5)
    df.loc[[0, 1], "score"] = np.nan
    h = allocate(df, top_n=8, max_weight=0.5)
    assert set(h["symbol"]) == {"T2", "T3", "T4"}
    assert allocate(df.assign(score=np.nan), top_n=3).empty


def test_allocate_rejects_unknown_method():
    with pytest.raises(ValueError):
        allocate(scored(3), method="risk_parity")
