# Methodology

This document describes exactly what the code in `src/` does. Default parameter values are the ones in `config.yaml`.

## Thesis

Companies that **grow measurably** (revenue, earnings and free cash flow) and that **do not trade at any price** (valuation relative to peers) make a healthier candidate list than whatever is popular at the moment. The system does not forecast prices. It **filters** and **ranks** by fixed rules, so the same inputs on the same date always produce the same output.

In data-science terms:

- **Estimation from short, noisy series:** multi-year growth from 3–4 annual observations with sign changes and gaps.
- **Data validation:** alias-based field matching, sanity bounds, and explicit flags instead of silent imputation.
- **Ranking under missing data:** cross-sectional percentile ranks, re-weighted over the components that exist.
- **Constrained allocation:** a Top-N selection with a per-name weight cap solved by water-filling.

## 1. Data (`src/data.py`)

Per ticker, `yfinance` supplies `info` (market cap, trailing P/E, sector, industry, currencies), the annual income statement, cash-flow statement and balance sheet, and one month of prices (latest close only).

Statement rows are found by alias (`ROW_ALIASES`), in two passes:

1. **Exact** label match (case and whitespace insensitive), in alias order.
2. **Whole-word** match as a fallback, so `ebit` never matches `EBITDA`. Labels such as "Cost of Revenue" are never taken as revenue.

FCF uses the reported "Free Cash Flow" row. If that row is absent, it is computed as operating cash flow + capital expenditure (capex is reported as a negative number).

Fetch errors never stop the run: a failed field becomes an empty frame or dict, and the name typically fails a gate later.

## 2. CAGR (`src/features.py`)

$$
\mathrm{CAGR}(V_{\text{start}}, V_{\text{end}}, t) = \left(\frac{V_{\text{end}}}{V_{\text{start}}}\right)^{1/t} - 1
$$

Rules:

1. Non-numeric and non-finite values are dropped.
2. The window keeps observations whose period-end date is within `max_cagr_years` years (plus 45 days of slack for fiscal-calendar drift) of the latest one.
3. `t` is the actual time between the two endpoint dates in years (days / 365.25). If the index is not date-like, consecutive annual columns are assumed (`t` = number of observations − 1).
4. At least 2 observations and `t ≥ 0.9` are required; otherwise the reason is `short_series`.
5. Both endpoints must be **strictly positive**:
   - opposite signs → `sign_change`
   - zero or both negative → `non_positive_endpoint`. With two negative endpoints the formula gives the wrong direction: −100 → −50 (losses halve) yields −16%, and −50 → −100 yields +19%.
6. CAGRs outside [−95%, +500%] are treated as data errors (`out_of_bounds`).

Intermediate years do not affect the CAGR: it is an endpoint measure and is sensitive to the base year. With about 3 years of Yahoo data this is a real limitation, documented in the README.

## 3. Ratios and data flags

| Feature | Formula | Set to missing when (flag) |
|---|---|---|
| `trailing_pe` | Yahoo `trailingPE` | missing (`pe_missing`), ≤ 0 (`pe_non_positive`), outside (1, 300] |
| `price_to_book` | market cap / latest stockholders' equity | equity ≤ 0 (`negative_equity`), currency mismatch, outside (0.05, 200] |
| `fcf_yield` | latest FCF / market cap | FCF ≤ 0 (`fcf_non_positive`), currency mismatch, outside (0, 0.5] |
| `operating_margin` | latest operating income / latest revenue | revenue ≤ 0, outside [−1, 1] |
| `debt_to_equity` | latest total debt / latest stockholders' equity | equity ≤ 0 (`negative_equity`), outside [0, 20] |

- P/B is computed rather than taken from Yahoo's `priceToBook`, which can be wrong (for BRK-B it was about 0.001, mixing A-share book value with the B-share price).
- `currency_mismatch` is raised when Yahoo's quote currency differs from the financial-statement currency. Market-cap-based ratios are then not computed.
- Out-of-bounds values are flagged `<feature>_out_of_bounds`. All bounds are configurable under `sanity_bounds`.

## 4. Eligibility gates (`src/scoring.py`)

A name must pass all gates to be ranked. Failures are listed in `gate_failures`.

| Gate | Default | Failure label |
|---|---|---|
| Excluded sector (exact name) | none | `excluded_sector` |
| Excluded industry (keyword, case-insensitive) | Banks, Insurance, Capital Markets, Asset Management, Mortgage, Financial Conglomerates, Shell Companies | `excluded_industry` |
| Usable growth CAGRs | ≥ 2 of 3 | `few_growth_metrics` |
| Operating margin | ≥ 8%; missing fails | `low_or_missing_margin` |
| Debt / equity | ≤ 2.5; missing passes | `high_leverage` |
| Negative equity | always fails when the D/E gate is on | `negative_equity` |
| Market cap | ≥ USD 10bn; missing fails | `small_or_missing_market_cap` |

**Why exclude banks and insurers?** Their yfinance "total revenue" and "free cash flow" are not comparable with industrial companies. An insurer's revenue includes investment gains and losses (BRK-B shows revenue jumping from about USD 234bn to 439bn between 2022 and 2023 for that reason), and a bank's operating cash flow is dominated by balance-sheet movements. Payment networks (Yahoo industry "Credit Services", e.g. V and MA) are kept. Setting `sectors: ["Financial Services"]` removes the whole sector instead.

## 5. Ranks and score

Percentile ranks (`pandas.Series.rank(pct=True)`, ties averaged) are computed **only among names that pass the gates**, so excluded names and their data problems cannot shift anyone else's rank. Lower-is-better features (P/E, P/B, D/E) are negated before ranking. Missing values stay missing.

- Growth `G` = mean of available ranks of revenue, earnings and FCF CAGR
- Value `V` = mean of available ranks of FCF yield, −P/E and −P/B
- Quality `Q` = mean of available ranks of operating margin and −D/E

$$
\text{score}_i = \frac{\sum_{k \in A_i} w_k C_{k,i}}{\sum_{k \in A_i} w_k}
$$

where `A_i` is the set of components available for name `i`, and the default weights are `w_G = 0.45`, `w_V = 0.35`, `w_Q = 0.20`. Missing components are re-weighted away, not scored as zero. A name with no component at all has no score.

## 6. From score to weights (`src/portfolio.py`)

1. Take the Top-N names by score (default 8).
2. Raw weights: `equal` (1 each) or `score` (the score itself).
3. Water-filling with cap `c`: normalise; pin any name above `c` at `c`; redistribute the remaining budget to unpinned names in proportion to their raw weights; repeat until no weight exceeds `c`. The result sums to 1 with every weight ≤ `c`.
4. If `N × c < 1` the cap cannot be met. It is relaxed to `1/N` (equal weight) and a `RuntimeWarning` is emitted, so the weights still sum to 1 rather than leaving an implicit cash position.

Equal weight is the default because, with 8–15 names and noisy retail data, estimated optimisers (e.g. mean-variance) tend to overfit.

## 7. Biases and mitigations

| Bias | Source here | Mitigation in this repo |
|---|---|---|
| Look-ahead | yfinance statements are the latest revision | No backtest; screens are dated snapshots only |
| Reporting lag | fundamentals are published after fiscal year-end | Documented; a historical study would need a lag of about 90 days |
| Survivorship | universe = currently listed names | Documented; no historical performance claims |
| Data snooping | thresholds and weights chosen by judgement | All parameters in `config.yaml`; sensitivity analysis is on the roadmap |
| Short history | about 3–4 annual columns | CAGR horizon reported per metric (`*_cagr_years`) |
| Bad vendor fields | e.g. Yahoo P/B for BRK-B | Computed ratios, sanity bounds, `data_flags` |

## 8. Tests

`tests/` covers CAGR edge cases (sign changes, negative and zero endpoints, gaps, short series, date-based horizons), row matching, feature guards (negative equity, non-positive P/E and FCF, bounds, currency mismatch), gates and exclusions, ranking with missing values, and the water-filling cap (feasible and infeasible). The tests use synthetic data and need no network.
