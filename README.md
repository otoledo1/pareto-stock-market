[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22608156.svg)](https://doi.org/10.5281/zenodo.22608156)

# The Pareto Principle Applied to the Stock Market

Replication code and data for *The Pareto Principle Applied to the Stock Market:
How a Few Trading Days Affect Financial Returns in Large-Cap Equities*
(Omer Toledo, The Nueva School; project advisor Kosrow Dehnad, Columbia University).

The paper asks how much of the widely quoted "missing the ten best days" effect
survives once it is benchmarked against a market with no fat tails. Across 100
large-cap S&P 500 stocks over 249 trading days, removing each stock's ten best
days costs a median of 42.92 percentage points and flips 74 of 100 stocks from a
winning year to a losing one.

Most of that magnitude is mechanical. A Gaussian benchmark matching each stock's
volatility, and its expected cumulative log return before positivity selection,
reproduces the effect's order of magnitude — so an investor quoting the
missing-best-days statistic as evidence of unusual market behavior is mostly
quoting order statistics. But that benchmark has a defect: its positivity
rejection step is one-sided, which leaves the simulated return level above the
data. A stricter benchmark that fixes each stock's *realized* annual return does
not reproduce the effect. Under every return-conditioned calibration convention
tested, the observed median gap exceeds all 2,000 replications and the
leave-top-10-out fit falls below all of them.

The sharpest result needs no calibration at all. An exact per-stock conditional
test puts 50 of 99 stocks beyond the 97.5th percentile of the Gaussian null, and
23 reject at a Bonferroni family-wise 5% level — a statement that holds whatever
the dependence across stocks.

## Reproducing the results

```bash
pip install -r requirements.txt
./run_all.sh
```

Tables land in `output/tables/`, figures in `output/figures/`. The full run takes
a few minutes; the Gaussian benchmarks and the block bootstrap dominate the
runtime. Every number in the manuscript regenerates from the archived panel under
the pinned environment and the fixed seed `20260712`.

## Monte Carlo conventions

Four implementation choices affect the simulated figures and are not recoverable
from the manuscript text alone. A replicator who guesses differently will get
different numbers.

1. **Return-conditioned null draw order.** One `(249, 99)` standard-normal block
   per replication, not 99 separate per-stock vectors. Per-stock draws shift the
   conditioned flip p-value from 0.174 to about 0.155.
2. **Exact per-stock test.** 500,000 × 249 draws, `ddof=1`. Monte Carlo p-values
   follow Phipson and Smyth (2010): `p = (b+1)/(B+1)`, so zero exceedances give
   `1/500001 ≈ 2e-6`. That is a floor imposed by the reference sample, not a
   certified bound on the continuous tail probability.
3. **Block bootstrap.** Non-circular moving blocks, `ceil(249/l)` blocks, starts
   uniform on `[0, 249-l]`, concatenated and truncated to 249 days. One generator
   is shared across block lengths 5, 10 and 21, drawn in that order, within each
   (scheme, screen) pass. The circular and stationary variants are also reported;
   the stationary bootstrap uses geometric block lengths of mean `l` with a
   circular wrap.
4. **Kurtosis.** Bias-corrected (`bias=False`). The biased estimator gives a
   median of 3.012 and 87 of 99 stocks above 1, against the reported 3.098 and 88.

Changing the seed will shift simulated medians and 95% ranges by small amounts
without changing any conclusion. Note that NumPy does **not** guarantee that
`default_rng` output is identical across versions, so exact regeneration needs the
pinned stack below — though in practice these figures have reproduced across
several (see *Environment*).

## Pipeline

Scripts run in order. Each reads from `data/` and writes to `output/`.

| Script | Produces |
| --- | --- |
| `01_fetch_prices.py` | Re-pulls prices from Yahoo Finance and applies the four-step selection, then compares the resulting ticker set against the archived sample. Not part of `run_all.sh`; see *Data provenance*. |
| `02_build_panel.py` | Normalizes the archived export into `data/panel_returns.csv` |
| `03_simulate.py` | Buy-and-hold vs. miss-top-10 per stock; summary statistics; flip rate by return quartile |
| `04_regressions.py` | Volatility-gap regressions with HC3 errors; influence diagnostics; robust slopes; leave-top-10-out regressor |
| `05_date_clustering.py` | Excluded-date overlap against a random-assignment null |
| `06_identity.py` | The exact gap decomposition; stand-alone return and wealth share; worst-day and both-tail exclusions |
| `07_gaussian_benchmark.py` | The drift-calibrated Gaussian null, 2,000 replications, including its own median buy-and-hold return; Equation (5.1) evaluated in closed form for all 99 stocks |
| `08_extensions.py` | Rolling six-month windows; exclusion thresholds k = 1 to 20; sector regressions |
| `09_figures.py` | The two manuscript figures |
| `10_return_conditioned_full.py` | The return-conditioned Gaussian null under four calibration conventions, all four headline statistics |
| `11_exact_conditional_test.py` | The exact per-stock conditional test; Bonferroni counts; dispersion of `S_i` |
| `12_block_bootstrap.py` | Block bootstrap over dates: three schemes, with and without the winner screen re-applied inside each resample |
| `13_correlated_null.py` | A return-conditioned null carrying the sample's cross-stock correlation matrix. **Run after 11** — it reads `output/tables/exact_conditional_summary.csv`. |

Shared constants live in `config.py`; helper functions in `src/common.py`.

The figures are written as `volatility_vs_gap.png` and `buy_hold_vs_miss.png`.
These are the same two images that appear in the manuscript as
`volatility_vs_loss_chart.png` and `top10_impact_chart.png` respectively; only
the filenames differ.

## Data

| File | Contents |
| --- | --- |
| `data/raw/sheets_export_daily_returns.csv` | The archived panel as exported, interleaving price and return columns |
| `data/panel_returns.csv` | 249 x 100 daily simple returns, generated by step 02 |
| `data/candidate_tickers.txt` | The 148 candidate tickers, ordered by S&P 500 index weight |
| `data/gics_sectors.csv` | Ticker to GICS sector mapping used in Tables 16 and 17 |

Berkshire Hathaway Class B appears as `BRK-B` in the candidate list, following
Yahoo Finance's ticker convention, and as BRK.B in the manuscript text.

## Data provenance

Prices were retrieved in June 2026 from Yahoo Finance via `yfinance`, using
split- and dividend-adjusted closes (`auto_adjust=True`) for 2025-06-01 to
2026-06-01, giving 250 price observations and 249 daily returns per ticker.
Because the adjustment covers dividends as well as splits, the returns analyzed
throughout are total returns.

The sample was built in four steps: start from 148 unique S&P 500 tickers
ordered by index weight from a June 2026 snapshot; download adjusted closes and
forward-fill; keep only tickers with a net-positive return over the window; keep
the 100 highest-weighted survivors. Forty-eight candidates are absent, at most
two of them removed by the final truncation. The original coverage and endpoint
records are not preserved, so the remaining 46 omissions cannot be assigned
conclusively to non-positive returns rather than to unavailable or incomplete
price data. Negative returns are the most likely account, but the sample is not
verified to be *every* net-positive name among the top 148 by weight; what is
established is that it is not a positive-return subset of a fixed top-100
universe.

**Yahoo Finance revises adjusted price history over time, so re-running
`01_fetch_prices.py` may not reproduce the archived panel exactly.** The
archived export in `data/raw/` is the object of record for every number in the
paper, and no later step reads step 01's output.

What step 01 does and does not verify is worth stating plainly. It re-derives
the sample selection and reports which tickers a fresh pull would add or drop
relative to the archived sample, so the construction rule can be checked
independently. It does not re-derive the returns: in the original workflow the
price-to-return conversion was done in a spreadsheet rather than in code, so a
fresh pull cannot be carried through the rest of the pipeline. Reproducing the
reported statistics from a new download would require that conversion step,
which is not included here.

Three limitations follow from the construction and are discussed in the paper.
The sample is winner-conditioned, since roughly a third of the candidate pool
was discarded for losing money. It embeds look-ahead bias, since index
membership and weights come from a June 2026 snapshot applied to returns
beginning in June 2025. And the candidate list was assembled by hand rather than
pulled from a timestamped vendor file, so its weight ordering is preserved in
`data/candidate_tickers.txt` but is not citable to a dated source.

## Environment

`requirements.txt` pins the versions used to produce the released output:
Python 3.13.15, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, statsmodels 0.15.0.
These are the versions named in the manuscript's Data and Code Availability
statement.

The Monte Carlo figures have also reproduced under NumPy 2.3.5 and 2.4.4, under
pandas 3.0.2, and under statsmodels 0.14.6, on three independent machines, so the
results are not tied to a single point release. NumPy makes no general guarantee
of `default_rng` stability across versions, so the pinned stack remains the
reference.

To record the versions in a different environment:

```bash
python -c "import importlib.metadata as m, sys; print('# Python ' + sys.version.split()[0]); [print(p + '==' + m.version(p)) for p in ['numpy','pandas','scipy','statsmodels','matplotlib','yfinance']]"
```

## Changelog

**v1.1** — second-round revision. Adds the return-conditioned Gaussian benchmark
under four calibration conventions (`10`), the exact per-stock conditional test
(`11`), the block bootstrap with scheme and winner-screen variants (`12`), and a
correlated-Gaussian null (`13`). `07` now covers the drift-calibrated null only,
and adds its median buy-and-hold return and the closed-form truncation identity;
the return-conditioned null it previously reported for the flip rate alone is
now handled in full by `10`. Monte Carlo conventions documented above;
environment re-pinned.

**v1.0** — initial release.

## Citation

Toledo, O. *The Pareto Principle Applied to the Stock Market: How a Few Trading
Days Affect Financial Returns in Large-Cap Equities.* Submitted to SIAM
Undergraduate Research Online, 2026.

