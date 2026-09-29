"""Fetch prices and financial statements from yfinance.

Yahoo data is convenient, not institutional. Fundamentals are the latest
available revision, not a point-in-time panel, and only 3-4 annual columns
are usually returned. Treat any historical use of these numbers as biased.
"""

from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
import yfinance as yf

# Statement row labels, in order of preference. Matching is exact first,
# then whole-word (see ``pick_row``).
ROW_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("total revenue", "operating revenue", "revenue"),
    "net_income": (
        "net income",
        "net income common stockholders",
        "net income from continuing operation net minority interest",
        "reconciled net income",
    ),
    "operating_income": ("operating income", "total operating income as reported", "ebit"),
    "fcf": ("free cash flow",),
    "operating_cf": ("operating cash flow", "total cash from operating activities"),
    "capex": ("capital expenditure", "capital expenditures"),
    "total_debt": ("total debt",),
    "equity": ("stockholders equity", "total stockholder equity", "common stock equity"),
}

# Labels that contain an alias as a whole word but mean something else,
# e.g. "cost of revenue" must never be picked as revenue.
_FALSE_FRIEND_PREFIXES = ("cost of", "other ", "non ", "total other")


def _norm(label: Any) -> str:
    return " ".join(str(label).strip().lower().split())


def pick_row(df: pd.DataFrame | None, aliases: tuple[str, ...]) -> pd.Series | None:
    """Return the first statement row matching ``aliases``.

    Pass 1: exact (case/whitespace-insensitive) label match, in alias order.
    Pass 2: whole-word match (so ``ebit`` never matches ``ebitda``), skipping
    labels such as ``cost of revenue``.
    """
    if df is None or df.empty:
        return None
    index_map = {_norm(i): i for i in df.index}

    for alias in aliases:
        if alias in index_map:
            return df.loc[index_map[alias]]

    for alias in aliases:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
        for key, original in index_map.items():
            if key.startswith(_FALSE_FRIEND_PREFIXES):
                continue
            if pattern.search(key):
                return df.loc[original]
    return None


def statement_to_annual(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return columns sorted oldest -> newest (fiscal period end dates)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = pd.to_datetime(out.columns, errors="coerce")
    out = out.loc[:, out.columns.notna()]
    return out.reindex(sorted(out.columns), axis=1)


def fetch_ticker(symbol: str, sleep_s: float = 0.35) -> dict[str, Any]:
    """Download one ticker. Returns a dict; never raises to the pipeline."""
    time.sleep(max(sleep_s, 0.0))
    ticker = yf.Ticker(symbol)

    def _safe(getter, default):
        try:
            value = getter()
            return default if value is None else value
        except Exception:
            return default

    info: dict[str, Any] = _safe(lambda: ticker.info, {})
    income = statement_to_annual(_safe(lambda: ticker.income_stmt, pd.DataFrame()))
    cash = statement_to_annual(_safe(lambda: ticker.cashflow, pd.DataFrame()))
    balance = statement_to_annual(_safe(lambda: ticker.balance_sheet, pd.DataFrame()))
    hist = _safe(lambda: ticker.history(period="1mo", auto_adjust=True), pd.DataFrame())

    return {
        "symbol": symbol,
        "info": info,
        "income": income,
        "cash": cash,
        "balance": balance,
        "history": hist,
    }


def fetch_universe(symbols: list[str], sleep_s: float = 0.35) -> list[dict[str, Any]]:
    return [fetch_ticker(symbol, sleep_s=sleep_s) for symbol in symbols]
