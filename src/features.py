"""Feature engineering: CAGR, quality ratios, valuation multiples.

Every feature is either a finite number or ``None``. When a value is dropped
(non-positive endpoint, out of sanity bounds, negative equity, ...) the reason
is recorded in the ``data_flags`` column so the screen stays auditable.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.data import ROW_ALIASES, pick_row

# Values outside these (inclusive) bounds are treated as data errors and set
# to None. Override in config.yaml under ``sanity_bounds``.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "cagr": (-0.95, 5.0),
    "trailing_pe": (1.0, 300.0),  # lower bound exclusive; non-positive P/E is always dropped
    "price_to_book": (0.05, 200.0),  # lower bound exclusive
    "fcf_yield": (0.0, 0.5),  # lower bound is exclusive: FCF must be > 0
    "operating_margin": (-1.0, 1.0),
    "debt_to_equity": (0.0, 20.0),
}

DAYS_PER_YEAR = 365.25
# Slack added to the look-back window so fiscal-year-end drift (e.g. 52/53-week
# years) does not drop the oldest column.
_WINDOW_SLACK_DAYS = 45


def cagr(start: float, end: float, years: float) -> float | None:
    """(end / start) ** (1 / years) - 1, only for strictly positive endpoints.

    Negative endpoints are rejected even when both are negative: going from
    -100 to -50 (losses shrinking) would otherwise read as -16% "growth".
    """
    try:
        start, end, years = float(start), float(end), float(years)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([start, end, years])):
        return None
    if years <= 0 or start <= 0 or end <= 0:
        return None
    value = (end / start) ** (1.0 / years) - 1.0
    return float(value) if np.isfinite(value) else None


def series_cagr(
    row: pd.Series | None,
    max_years: float = 4,
    min_years: float = 0.9,
) -> dict[str, Any]:
    """CAGR from an annual statement row (columns oldest -> newest).

    ``years`` comes from the period-end dates of the two endpoints, not from
    the number of observations, so a missing middle year does not shorten the
    horizon. If the index is not datetime-like, consecutive annual columns are
    assumed. Reasons when no CAGR is returned: ``missing_row``,
    ``short_series``, ``sign_change`` (endpoints of opposite sign),
    ``non_positive_endpoint`` (an endpoint is <= 0 without a sign flip).
    """
    payload: dict[str, Any] = {
        "cagr": None,
        "years": None,
        "n_obs": 0,
        "start": None,
        "end": None,
        "reason": None,
    }
    if row is None or row.empty:
        payload["reason"] = "missing_row"
        return payload

    values = pd.to_numeric(row, errors="coerce")
    values = values[np.isfinite(values.astype(float))].dropna()
    if len(values) < 2:
        payload["n_obs"] = int(len(values))
        payload["reason"] = "short_series"
        return payload

    if isinstance(values.index, pd.DatetimeIndex):
        dates = values.index
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates = pd.to_datetime(pd.Index(values.index), errors="coerce")
    if dates.notna().all():
        values = pd.Series(values.to_numpy(), index=dates).sort_index()
        horizon = pd.Timedelta(days=max_years * DAYS_PER_YEAR + _WINDOW_SLACK_DAYS)
        values = values[values.index >= values.index[-1] - horizon]
        years = (values.index[-1] - values.index[0]).days / DAYS_PER_YEAR if len(values) else 0.0
    else:
        values = values.iloc[-(int(max_years) + 1) :]
        years = float(len(values) - 1)

    payload["n_obs"] = int(len(values))
    if len(values) < 2 or years < min_years:
        payload["reason"] = "short_series"
        return payload

    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    payload.update({"start": start, "end": end, "years": round(float(years), 2)})

    if start * end < 0:
        payload["reason"] = "sign_change"
    elif start <= 0 or end <= 0:
        payload["reason"] = "non_positive_endpoint"
    else:
        payload["cagr"] = cagr(start, end, years)
        if payload["cagr"] is None:
            payload["reason"] = "undefined"
    return payload


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _safe_div(num: Any, den: Any) -> float | None:
    n, d = _to_float(num), _to_float(den)
    if n is None or d is None or d == 0:
        return None
    return n / d


def last_value(row: pd.Series | None) -> float | None:
    if row is None or row.empty:
        return None
    values = pd.to_numeric(row, errors="coerce").dropna()
    return _to_float(values.iloc[-1]) if not values.empty else None


def _bounded(
    name: str,
    value: float | None,
    bounds: dict[str, tuple[float, float]],
    flags: list[str],
    exclusive_low: bool = False,
) -> float | None:
    if value is None:
        return None
    low, high = bounds[name]
    too_low = value <= low if exclusive_low else value < low
    if too_low or value > high:
        flags.append(f"{name}_out_of_bounds")
        return None
    return value


def extract_features(
    bundle: dict[str, Any],
    max_years: float = 4,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    flags: list[str] = []
    info = bundle.get("info") or {}
    income = bundle.get("income")
    cash = bundle.get("cash")
    balance = bundle.get("balance")
    history = bundle.get("history")

    if not info:
        flags.append("missing_info")

    revenue = pick_row(income, ROW_ALIASES["revenue"])
    net_income = pick_row(income, ROW_ALIASES["net_income"])
    operating_income = pick_row(income, ROW_ALIASES["operating_income"])
    fcf = pick_row(cash, ROW_ALIASES["fcf"])
    if fcf is None:
        ocf = pick_row(cash, ROW_ALIASES["operating_cf"])
        capex = pick_row(cash, ROW_ALIASES["capex"])
        if ocf is not None and capex is not None:
            # CapEx is reported as a negative number: FCF = OCF + CapEx.
            aligned = pd.concat([ocf, capex], axis=1, keys=["ocf", "capex"]).dropna()
            fcf = aligned["ocf"] + aligned["capex"] if not aligned.empty else None

    debt = pick_row(balance, ROW_ALIASES["total_debt"])
    equity = pick_row(balance, ROW_ALIASES["equity"])

    growth = {
        "revenue": series_cagr(revenue, max_years=max_years),
        "earnings": series_cagr(net_income, max_years=max_years),
        "fcf": series_cagr(fcf, max_years=max_years),
    }
    for res in growth.values():
        if res["cagr"] is not None:
            res["cagr"] = _bounded("cagr", res["cagr"], bounds, flags)
            if res["cagr"] is None:
                res["reason"] = "out_of_bounds"

    last_rev = last_value(revenue)
    last_oi = last_value(operating_income)
    last_fcf = last_value(fcf)
    last_debt = last_value(debt)
    last_equity = last_value(equity)

    price = None
    if history is not None and not getattr(history, "empty", True) and "Close" in history:
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if not close.empty:
            price = float(close.iloc[-1])

    market_cap = _to_float(info.get("marketCap"))
    if market_cap is not None and market_cap <= 0:
        market_cap = None

    # Market cap is in the quote currency, statements in the reporting
    # currency. Do not mix them (common for ADRs).
    quote_ccy = info.get("currency")
    report_ccy = info.get("financialCurrency") or quote_ccy
    ccy_ok = quote_ccy is None or report_ccy is None or quote_ccy == report_ccy
    if not ccy_ok:
        flags.append("currency_mismatch")

    # Balance sheet: negative (or zero) equity makes D/E and P/B meaningless.
    negative_equity = last_equity is not None and last_equity <= 0
    debt_to_equity = price_to_book = None
    if negative_equity:
        flags.append("negative_equity")
    elif last_equity is not None:
        debt_to_equity = _safe_div(last_debt, last_equity)
        if ccy_ok:
            price_to_book = _safe_div(market_cap, last_equity)

    trailing_pe = _to_float(info.get("trailingPE"))
    if trailing_pe is None:
        flags.append("pe_missing")
    elif trailing_pe <= 0:
        flags.append("pe_non_positive")
        trailing_pe = None

    fcf_yield = None
    if last_fcf is not None and last_fcf <= 0:
        flags.append("fcf_non_positive")
    elif ccy_ok:
        fcf_yield = _safe_div(last_fcf, market_cap)

    operating_margin = _safe_div(last_oi, last_rev) if (last_rev or 0) > 0 else None

    latest_period = None
    if income is not None and not income.empty:
        latest_period = str(pd.Timestamp(income.columns[-1]).date())

    return {
        "symbol": bundle.get("symbol"),
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": quote_ccy,
        "latest_fiscal_period": latest_period,
        "price": price,
        "market_cap": market_cap,
        "revenue_cagr": growth["revenue"]["cagr"],
        "revenue_cagr_years": growth["revenue"]["years"],
        "revenue_cagr_reason": growth["revenue"]["reason"],
        "earnings_cagr": growth["earnings"]["cagr"],
        "earnings_cagr_years": growth["earnings"]["years"],
        "earnings_cagr_reason": growth["earnings"]["reason"],
        "fcf_cagr": growth["fcf"]["cagr"],
        "fcf_cagr_years": growth["fcf"]["years"],
        "fcf_cagr_reason": growth["fcf"]["reason"],
        "operating_margin": _bounded("operating_margin", operating_margin, bounds, flags),
        "debt_to_equity": _bounded("debt_to_equity", debt_to_equity, bounds, flags),
        "negative_equity": bool(negative_equity),
        "trailing_pe": _bounded("trailing_pe", trailing_pe, bounds, flags, exclusive_low=True),
        "price_to_book": _bounded("price_to_book", price_to_book, bounds, flags, exclusive_low=True),
        "fcf_yield": _bounded("fcf_yield", fcf_yield, bounds, flags, exclusive_low=True),
        "usable_growth_metrics": sum(res["cagr"] is not None for res in growth.values()),
        "data_flags": ";".join(flags),
    }


def features_frame(
    bundles: list[dict[str, Any]],
    max_years: float = 4,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    rows = [extract_features(b, max_years=max_years, bounds=bounds) for b in bundles]
    return pd.DataFrame(rows)
