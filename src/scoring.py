"""Eligibility gates and composite growth / value / quality score.

Ranks are percentile ranks computed cross-sectionally among the names that
pass the gates in the current run. This is a relative screen, not an
absolute valuation model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROWTH_HIGHER_BETTER = ["revenue_cagr", "earnings_cagr", "fcf_cagr"]
VALUE_HIGHER_BETTER = ["fcf_yield"]
VALUE_LOWER_BETTER = ["trailing_pe", "price_to_book"]
QUALITY_HIGHER_BETTER = ["operating_margin"]
QUALITY_LOWER_BETTER = ["debt_to_equity"]

DEFAULT_WEIGHTS = {"growth": 0.45, "value": 0.35, "quality": 0.20}


def rank_01(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Percentile rank in (0, 1]; missing values stay missing."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if not higher_is_better:
        s = -s
    return s.rank(pct=True, method="average")


def _matches_any(text: object, keywords: list[str]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def apply_gates(df: pd.DataFrame, gates: dict, exclusions: dict | None = None) -> pd.DataFrame:
    """Return a frame with ``passes_gates`` (bool) and ``gate_failures`` (str)."""
    exclusions = exclusions or {}
    failures: dict[str, list[str]] = {idx: [] for idx in df.index}

    def fail(mask: pd.Series, label: str) -> None:
        for idx in df.index[mask.fillna(False).astype(bool)]:
            failures[idx].append(label)

    sectors = [s.lower() for s in exclusions.get("sectors", []) or []]
    if sectors and "sector" in df:
        fail(df["sector"].fillna("").str.lower().isin(sectors), "excluded_sector")

    keywords = exclusions.get("industry_keywords", []) or []
    if keywords and "industry" in df:
        fail(df["industry"].apply(lambda x: _matches_any(x, keywords)), "excluded_industry")

    min_metrics = int(gates.get("min_usable_growth_metrics", 2))
    fail(df["usable_growth_metrics"].fillna(0) < min_metrics, "few_growth_metrics")

    min_margin = gates.get("min_operating_margin")
    if min_margin is not None:
        # Missing margin fails: we cannot verify profitability.
        fail(~(df["operating_margin"] >= float(min_margin)), "low_or_missing_margin")

    max_de = gates.get("max_debt_to_equity")
    if max_de is not None:
        # Missing D/E passes (e.g. no debt row); negative equity always fails.
        fail(df["debt_to_equity"] > float(max_de), "high_leverage")
        if "negative_equity" in df:
            fail(df["negative_equity"].fillna(False).astype(bool), "negative_equity")

    min_mcap = gates.get("min_market_cap")
    if min_mcap is not None:
        fail(~(df["market_cap"] >= float(min_mcap)), "small_or_missing_market_cap")

    out = pd.DataFrame(index=df.index)
    out["gate_failures"] = pd.Series({k: ";".join(v) for k, v in failures.items()})
    out["passes_gates"] = out["gate_failures"] == ""
    return out


def _component(frame: pd.DataFrame, higher: list[str], lower: list[str]) -> pd.Series:
    parts = [rank_01(frame[c], True) for c in higher if c in frame]
    parts += [rank_01(frame[c], False) for c in lower if c in frame]
    if not parts:
        return pd.Series(np.nan, index=frame.index)
    # Mean over the sub-metrics that exist for each name.
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def composite(ranks: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted mean of component ranks, re-weighted over available components.

    If a name has no value metrics at all, its score is the weighted mean of
    growth and quality only (weights renormalised to sum to 1 for that name).
    Names with no component at all get NaN.
    """
    w = pd.Series({k: float(weights.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS})
    cols = [f"{k}_rank" for k in w.index]
    r = ranks[cols].astype(float)
    wmat = pd.DataFrame(np.tile(w.to_numpy(), (len(r), 1)), index=r.index, columns=cols)
    wmat = wmat.where(r.notna(), 0.0)
    total = wmat.sum(axis=1)
    num = (r.fillna(0.0) * wmat).sum(axis=1)
    return (num / total).where(total > 0)


def score(
    df: pd.DataFrame,
    weights: dict | None = None,
    gates: dict | None = None,
    exclusions: dict | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out = out.join(apply_gates(out, gates or {}, exclusions))

    eligible = out[out["passes_gates"]]
    ranks = pd.DataFrame(index=out.index, columns=["growth_rank", "value_rank", "quality_rank"], dtype=float)
    if not eligible.empty:
        ranks.loc[eligible.index, "growth_rank"] = _component(eligible, GROWTH_HIGHER_BETTER, [])
        ranks.loc[eligible.index, "value_rank"] = _component(eligible, VALUE_HIGHER_BETTER, VALUE_LOWER_BETTER)
        ranks.loc[eligible.index, "quality_rank"] = _component(
            eligible, QUALITY_HIGHER_BETTER, QUALITY_LOWER_BETTER
        )
    out = out.join(ranks)
    out["score"] = composite(out, weights or {})
    out.loc[~out["passes_gates"], "score"] = np.nan
    return out.sort_values(["score", "symbol"], ascending=[False, True], na_position="last")
