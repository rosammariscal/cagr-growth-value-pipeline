# Systematic Growth/Value Equity Screen

A small, reproducible Python pipeline that turns public market data and annual financial statements (via [`yfinance`](https://github.com/ranaroussi/yfinance)) into a **composite growth + value + quality score**, applies explicit eligibility gates, and builds a capped Top-N weighting.

This is a **personal research project**. It is a screening tool, not a trading system, and it makes **no performance claims**: there is no backtest and no track record in this repository.

```
fetch (yfinance) → features (CAGR, ratios) → sanity checks → gates → cross-sectional ranks → composite score → capped Top-N weights → CSV
```

## Problem

A growth/value screen has to answer three questions with rules rather than opinions:

1. **Is growth real and measurable?** Multi-year growth of revenue, net income and free cash flow, computed only when the series supports it.
2. **Is the price reasonable relative to peers?** Valuation multiples ranked within the universe, so high growth at any price is penalised.
3. **Is the business sound enough to rank at all?** Hard gates on margin, leverage, size and data usability.

The engineering problem underneath is typical of applied data science: incomplete and inconsistent source data, features with business-rule edge cases, ranking under missing values, and a constrained allocation step, all made reproducible and testable.

## Method

### 1. Growth features (CAGR)

For revenue, net income and free cash flow (FCF; reported, or operating cash flow + capex as a fallback):

$$
\mathrm{CAGR} = \left(\frac{V_{\text{end}}}{V_{\text{start}}}\right)^{1/t} - 1,
\qquad t = \frac{\text{date}_{\text{end}} - \text{date}_{\text{start}}}{365.25\ \text{days}}
$$

- `t` comes from the **fiscal period-end dates** of the two endpoints, not from the number of observations. A missing middle year does not shorten the horizon.
- The window is the most recent `max_cagr_years` (default 4) years. Yahoo usually returns only 3–4 annual columns, so in practice `t ≈ 3`.
- A CAGR is computed **only if both endpoints are strictly positive**. Otherwise it is left missing, with a reason:
  - `sign_change`: endpoints have opposite signs (loss → profit or profit → loss).
  - `non_positive_endpoint`: an endpoint is zero, or both are negative. Going from −100 to −50 would otherwise read as −16% "growth" while losses shrink.
  - `short_series`: fewer than two usable annual values (or under ~0.9 years apart).
  - `missing_row`: the statement line is not available.

### 2. Value and quality features

| Feature | Definition | Direction | Dropped when |
|---|---|---|---|
| Trailing P/E | from Yahoo `trailingPE` | lower is better | ≤ 0 or missing |
| P/B | market cap / latest stockholders' equity | lower is better | equity ≤ 0 |
| FCF yield | latest FCF / market cap | higher is better | FCF ≤ 0 |
| Operating margin | latest operating income / revenue | higher is better | revenue ≤ 0 |
| Debt / equity | latest total debt / stockholders' equity | lower is better | equity ≤ 0 (flagged `negative_equity`) |

P/B and FCF yield are also dropped when the quote currency differs from the reporting currency (common for ADRs). **Sanity bounds** (`sanity_bounds` in `config.yaml`) turn implausible values into missing values, e.g. P/B outside (0.05, 200] or CAGR outside [−95%, +500%]. Every dropped value is recorded in the `data_flags` column.

### 3. Eligibility gates

A name is ranked only if it passes all gates (defaults in `config.yaml`):

- not in an excluded industry (banks, insurance, capital markets, asset management, mortgage; see *Limitations*)
- at least 2 of the 3 growth CAGRs are usable
- operating margin ≥ 8% (a missing margin fails)
- debt/equity ≤ 2.5 (a missing value passes; negative equity fails)
- market cap ≥ USD 10bn

Failed gates are listed per name in the `gate_failures` column.

### 4. Ranks and composite score

Within the names that pass the gates, each feature becomes a percentile rank in (0, 1]. Each component is the mean of the ranks available for that name:

$$
G = \operatorname{mean}(r_{\text{rev}}, r_{\text{ni}}, r_{\text{fcf}}),\quad
V = \operatorname{mean}(r_{\text{fcf yield}}, r_{-\text{P/E}}, r_{-\text{P/B}}),\quad
Q = \operatorname{mean}(r_{\text{margin}}, r_{-\text{D/E}})
$$

$$
\text{score} = \frac{\sum_{k \in \text{avail}} w_k C_k}{\sum_{k \in \text{avail}} w_k},
\qquad w_G = 0.45,\ w_V = 0.35,\ w_Q = 0.20
$$

A missing component is **re-weighted away** rather than treated as zero, so missing data neither rewards nor zeroes a name.

### 5. Allocation

The Top-N names by score (default N = 8) get `equal` or `score`-proportional weights, then an **iterative water-filling cap**: any weight above `max_weight` is pinned at the cap and the excess is redistributed pro rata to the uncapped names, repeating until none exceeds the cap. Weights always sum to 1. If the cap is infeasible (N × cap < 1), it is relaxed to 1/N (equal weight) with a warning.

## Results: screen snapshot (2026-09-29)

This is real output from `python -m src.pipeline --config config.yaml --snapshot-dir reports`, run on 2026-09-29 against the demo universe of 15 US large caps in `config.yaml`. Full outputs: [`reports/screen_2026-09-29.csv`](reports/screen_2026-09-29.csv) and [`reports/holdings_2026-09-29.csv`](reports/holdings_2026-09-29.csv).

- **Universe:** 15 tickers. **Passed gates:** 9.
- **Excluded:** AMZN (fewer than 2 usable CAGRs: net income and FCF change sign over the window), BRK-B and JPM (excluded industries), COST and UNH (operating margin < 8%), HD (debt/equity > 2.5).
- Growth CAGRs cover about 3 years (latest fiscal year ending between Aug 2025 and Jun 2026, depending on the company).

| Rank | Ticker | Score | Growth rank | Value rank | Quality rank | Weight |
|---:|---|---:|---:|---:|---:|---:|
| 1 | META | 0.778 | 0.85 | 0.78 | 0.61 | 12.5% |
| 2 | NVDA | 0.769 | 1.00 | 0.37 | 0.94 | 12.5% |
| 3 | MSFT | 0.619 | 0.48 | 0.70 | 0.78 | 12.5% |
| 4 | GOOGL | 0.581 | 0.56 | 0.63 | 0.56 | 12.5% |
| 5 | MA | 0.520 | 0.56 | 0.52 | 0.44 | 12.5% |
| 6 | AVGO | 0.517 | 0.78 | 0.22 | 0.44 | 12.5% |
| 7 | V | 0.500 | 0.33 | 0.56 | 0.78 | 12.5% |
| 8 | PEP | 0.491 | 0.30 | 0.93 | 0.17 | 12.5% |

Ranks are relative to the 9 eligible names only. This table shows how the rules sort one small universe on one date. It is not a recommendation and says nothing about future returns.

## Limitations and biases

- **Not point-in-time.** `yfinance` returns the *current* version of historical statements, including restatements, not what was known on each past date. Any historical simulation built on it has look-ahead bias, which is why this repo contains no backtest.
- **Reporting lag.** Annual fundamentals are published weeks to months after fiscal year-end. A proper historical study would need an explicit lag (e.g. 90 days).
- **Short history.** Yahoo typically provides 3–4 annual columns, so the "multi-year" CAGR is about 3 years and sensitive to the base year.
- **Survivorship.** The universe is companies that trade today. Delisted names are invisible.
- **Data snooping.** Gate thresholds and component weights were set by judgement, not estimated. Changing them changes the ranking, so treat them as parameters to stress-test.
- **Financials are not comparable.** For banks, insurers and conglomerates, yfinance "revenue" and "FCF" do not mean the same thing as for industrial companies. BRK-B's reported total revenue includes investment gains and losses, and a bank's FCF is essentially operating cash flow driven by balance-sheet flows. By default such industries are excluded via `exclusions.industry_keywords`. Payment networks (V, MA; Yahoo industry "Credit Services") are kept because their statements are standard. Set `exclusions.sectors: ["Financial Services"]` to drop the whole sector.
- **Data quality.** Yahoo fields can be wrong (e.g. Yahoo's own `priceToBook` for BRK-B was about 0.001). This is why P/B is computed from market cap and equity, and implausible values are bounded and flagged rather than silently ranked.
- **Relative, not absolute.** Percentile ranks say which names look better *within this universe*, not whether any of them is cheap.

## Project structure

```
cagr-growth-value-pipeline/
├── config.yaml            # universe, gates, weights, exclusions, sanity bounds
├── src/
│   ├── data.py            # yfinance fetch, statement row matching
│   ├── features.py        # CAGR, ratios, sanity bounds, data flags
│   ├── scoring.py         # gates, percentile ranks, composite score
│   ├── portfolio.py       # Top-N selection, water-filling cap
│   └── pipeline.py        # CLI entry point
├── tests/                 # pytest unit tests (no network)
├── reports/               # dated screen snapshots
├── docs/METHODOLOGY.md    # longer write-up of the rules
└── data/                  # raw/ and processed/ outputs (git-ignored)
```

## Stack

Python 3.10+ · pandas · NumPy · yfinance · PyYAML · pytest

## How to run

```bash
git clone https://github.com/rosammariscal/cagr-growth-value-pipeline.git
cd cagr-growth-value-pipeline
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m pytest                    # unit tests, no network needed
PYTHONPATH=. python -m src.pipeline --config config.yaml
# optional: also save dated copies to reports/
PYTHONPATH=. python -m src.pipeline --config config.yaml --snapshot-dir reports
```

Yahoo rate-limits aggressive clients. The fetcher pauses `sleep_s` seconds between tickers, so keep universes modest.

### Outputs

- `data/processed/screen.csv`: every ticker with all features, CAGR reasons, `data_flags`, `passes_gates`, `gate_failures`, component ranks and score.
- `data/processed/holdings.csv`: the Top-N rows with a `weight` column.
- With `--snapshot-dir DIR`: dated copies `DIR/screen_YYYY-MM-DD.csv` and `DIR/holdings_YYYY-MM-DD.csv`.

Every row has an `as_of` date. `data/processed/` is git-ignored; only deliberate snapshots in `reports/` are committed.

## Roadmap

- Point-in-time fundamentals, or at least an explicit reporting lag, before any historical evaluation
- Sensitivity analysis of the ranking to gate thresholds and component weights
- Broader universes (e.g. Mexican BMV listings) with FX conversion to a base currency
- Growth-consistency features (e.g. dispersion of year-over-year growth)

## Disclaimer

This is a personal research and data-science project. It is **not investment advice**, not a recommendation to buy or sell any security, and **not an audited track record**. The author does not manage client money or offer advisory services. Outputs depend on third-party data that may be incomplete or wrong.

## License

[MIT](LICENSE) © 2026 Rosa María Mariscal Romero
