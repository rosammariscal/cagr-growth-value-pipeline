"""CLI: fetch -> features -> gates & score -> allocate -> write CSVs."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from src.data import fetch_universe
from src.features import features_frame
from src.portfolio import allocate
from src.scoring import score

SUMMARY_COLS = [
    "symbol",
    "name",
    "score",
    "growth_rank",
    "value_rank",
    "quality_rank",
    "revenue_cagr",
    "earnings_cagr",
    "fcf_cagr",
    "trailing_pe",
    "weight",
]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: str = "config.yaml", snapshot_dir: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    as_of = dt.date.today().isoformat()
    symbols = list(cfg["universe"])

    bundles = fetch_universe(symbols, sleep_s=float(cfg.get("sleep_s", 0.35)))
    bounds = {k: tuple(v) for k, v in (cfg.get("sanity_bounds") or {}).items()}
    feats = features_frame(bundles, max_years=float(cfg.get("max_cagr_years", 4)), bounds=bounds)
    scored = score(feats, cfg.get("weights", {}), cfg.get("gates", {}), cfg.get("exclusions", {}))
    holdings = allocate(
        scored,
        top_n=int(cfg.get("top_n", 10)),
        max_weight=float(cfg.get("max_weight", 0.15)),
        method=str(cfg.get("allocation", "equal")),
    )
    scored.insert(0, "as_of", as_of)
    holdings.insert(0, "as_of", as_of)

    out_dir = Path(cfg.get("output_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "screen.csv", out_dir / "holdings.csv"]
    scored.to_csv(written[0], index=False)
    holdings.to_csv(written[1], index=False)

    if snapshot_dir:
        snap = Path(snapshot_dir)
        snap.mkdir(parents=True, exist_ok=True)
        written += [snap / f"screen_{as_of}.csv", snap / f"holdings_{as_of}.csv"]
        scored.to_csv(written[2], index=False)
        holdings.to_csv(written[3], index=False)

    n_pass = int(scored["passes_gates"].sum())
    print(f"\nAs of {as_of}: {len(scored)} tickers screened, {n_pass} passed gates.")
    failed = scored.loc[~scored["passes_gates"], ["symbol", "gate_failures"]]
    for _, r in failed.iterrows():
        print(f"  excluded {r['symbol']:<6} {r['gate_failures']}")
    print("\nHoldings")
    print(holdings[[c for c in SUMMARY_COLS if c in holdings.columns]].to_string(index=False))
    print(f"\nWeights sum to {holdings['weight'].sum():.4f}; max weight {holdings['weight'].max():.4f}")
    print("Wrote " + ", ".join(str(p) for p in written))
    return holdings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the growth/value screen")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Also write dated copies (screen_YYYY-MM-DD.csv, holdings_YYYY-MM-DD.csv) here",
    )
    args = parser.parse_args()
    run(args.config, snapshot_dir=args.snapshot_dir)


if __name__ == "__main__":
    main()
