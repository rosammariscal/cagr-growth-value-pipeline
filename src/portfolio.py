"""Turn a scored universe into long-only position weights with a cap."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def water_fill(raw: pd.Series, cap: float) -> pd.Series:
    """Normalise ``raw`` to sum to 1 with no weight above ``cap``.

    Iterative water-filling: names above the cap are pinned at the cap and the
    remaining budget is redistributed pro rata to the rest, until no weight
    exceeds the cap. If ``len(raw) * cap < 1`` the cap is infeasible; it is
    relaxed to ``1 / len(raw)`` (i.e. equal weight) and a warning is issued,
    so weights still sum to 1.
    """
    raw = pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    n = len(raw)
    if n == 0:
        return raw
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=raw.index)

    cap = float(cap)
    if n * cap < 1.0 - 1e-12:
        warnings.warn(
            f"max_weight={cap:.4f} is infeasible for {n} names (n * cap < 1); "
            f"relaxing cap to 1/n = {1.0 / n:.4f} and renormalising.",
            RuntimeWarning,
            stacklevel=2,
        )
        cap = 1.0 / n

    w = raw / raw.sum()
    pinned = pd.Series(False, index=raw.index)
    for _ in range(n):
        over = (w > cap + 1e-12) & ~pinned
        if not over.any():
            break
        pinned |= over
        free = ~pinned
        budget = 1.0 - cap * pinned.sum()
        w[pinned] = cap
        free_raw = raw[free]
        if free.any():
            w[free] = (free_raw / free_raw.sum() if free_raw.sum() > 0 else 1.0 / free.sum()) * budget
    return w


def allocate(
    scored: pd.DataFrame,
    top_n: int = 10,
    max_weight: float = 0.15,
    method: str = "equal",
) -> pd.DataFrame:
    """Top-N names by score, weighted ``equal`` or pro rata to ``score``."""
    if method not in {"equal", "score"}:
        raise ValueError(f"allocation must be 'equal' or 'score', got {method!r}")
    eligible = (
        scored.dropna(subset=["score"])
        .sort_values("score", ascending=False)
        .head(int(top_n))
        .copy()
    )
    if eligible.empty:
        eligible["weight"] = pd.Series(dtype=float)
        return eligible

    raw = eligible["score"].astype(float) if method == "score" else pd.Series(1.0, index=eligible.index)
    eligible["weight"] = water_fill(raw, max_weight).to_numpy()
    assert np.isclose(eligible["weight"].sum(), 1.0)
    return eligible
