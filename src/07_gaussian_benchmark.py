"""Step 7. The drift-calibrated Gaussian benchmark, and why it fails.

Tables 8 and 10 (drift column). Simulates a market with no fat tails, matched
to each stock's own volatility and annual return, then applies the identical
miss-top-10 procedure. This measures how much of the observed effect is
produced by order statistics alone.

The null matches each stock's annual return only in expectation and then
redraws any path that ends non-positive, mirroring the sample's positive-return
filter. That rejection step is one-sided, so the surviving paths are
systematically richer than the target and the comparison is biased toward
finding no effect. The script quantifies the bias two ways: it records the
null's own median buy-and-hold return alongside the four headline statistics,
and it evaluates the truncation identity of Equation (5.1) in closed form.

Note on Equation (5.1): it gives an expected *log* return. Exponentiating it
does not give the expected compounded simple return, which is

    E[e^X - 1 | X > 0] = exp(m + v/2) Phi((m + v)/sqrt(v)) / Phi(m/sqrt(v)) - 1.

Both are reported below. The paper should label the exponentiated-log figures
as geometric-growth equivalents, or quote the arithmetic ones.

The return-conditioned null, which fixes each stock's realized annual return
rather than matching it in expectation, is step 10. It is the benchmark of
record and is no longer duplicated here.

Writes: output/tables/gaussian_benchmark.csv, benchmark_replications.csv,
        truncation_identity.csv
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from common import (config, drop_outlier, load_panel, percentile_of, r_squared,
                    stock_statistics, summarize, write)

panel = load_panel()
sessions = len(panel)
excl = drop_outlier(stock_statistics(panel))
n_stocks = len(excl)

daily_sd = (excl.volatility.values / 100) / np.sqrt(config.TRADING_DAYS)
drift = np.log(1 + excl.buy_hold.values / 100) / sessions

observed = {
    "median_gap": np.median(excl.gap.values),
    "r_squared": r_squared(excl.volatility.values, excl.gap.values),
    "flip_rate": 100 * (excl.miss_top_k.values < 0).mean(),
    "leaveout_r_squared": r_squared(excl.volatility_leaveout.values, excl.gap.values),
    "median_buy_hold": np.median(excl.buy_hold.values),
}

print(f"Simulating {config.N_BENCHMARK_REPS} cross-sections of {n_stocks} stocks "
      f"x {sessions} sessions, seed {config.SEED}")

rng = np.random.default_rng(config.SEED)
drift_null = {key: [] for key in observed}

for _ in range(config.N_BENCHMARK_REPS):
    gaps = np.empty(n_stocks)
    leaveout_vol = np.empty(n_stocks)
    buy_holds = np.empty(n_stocks)
    flips = 0
    for i in range(n_stocks):
        for _attempt in range(config.MAX_DRAW_ATTEMPTS):
            log_returns = rng.normal(drift[i], daily_sd[i], sessions)
            if log_returns.sum() > 0:
                break
        returns = np.expm1(log_returns)
        top = np.argsort(returns)[-config.K:]
        kept = np.delete(returns, top)
        buy_hold = np.prod(1 + returns) - 1
        missed = np.prod(1 + kept) - 1
        gaps[i] = (buy_hold - missed) * 100
        buy_holds[i] = buy_hold * 100
        flips += missed < 0
        leaveout_vol[i] = kept.std(ddof=1) * np.sqrt(config.TRADING_DAYS) * 100
    drift_null["median_gap"].append(np.median(gaps))
    drift_null["r_squared"].append(r_squared(excl.volatility.values, gaps))
    drift_null["flip_rate"].append(100 * flips / n_stocks)
    drift_null["leaveout_r_squared"].append(r_squared(leaveout_vol, gaps))
    drift_null["median_buy_hold"].append(np.median(buy_holds))

rows = []
for key in observed:
    rows.append({"quantity": key, "null": "drift-calibrated", "actual": observed[key],
                 **summarize(drift_null[key]),
                 "actual_percentile": percentile_of(drift_null[key], observed[key])})
write(pd.DataFrame(rows).set_index(["quantity", "null"]).round(4), "gaussian_benchmark")

replications = pd.DataFrame(drift_null)
replications.to_csv(config.TABLES / "benchmark_replications.csv", index=False)
print("-> output/tables/benchmark_replications.csv\n")

# ---- Equation (5.1): the one-sided rejection step, in closed form -----------
mean_log = np.log1p(excl.buy_hold.values / 100)
variance = (daily_sd ** 2) * sessions
sd_log = np.sqrt(variance)
ratio = mean_log / sd_log

truncation = pd.DataFrame({
    "buy_hold_pct": excl.buy_hold.values,
    "prob_non_positive_pct": 100 * scipy_stats.norm.cdf(-ratio),
    "expected_log_return_given_positive":
        mean_log + sd_log * scipy_stats.norm.pdf(ratio) / scipy_stats.norm.cdf(ratio),
    "geometric_equivalent_pct": 100 * (np.expm1(
        mean_log + sd_log * scipy_stats.norm.pdf(ratio) / scipy_stats.norm.cdf(ratio))),
    "arithmetic_expected_return_pct": 100 * (
        np.exp(mean_log + variance / 2)
        * scipy_stats.norm.cdf((mean_log + variance) / sd_log)
        / scipy_stats.norm.cdf(ratio) - 1),
    "unconditional_expected_return_pct": 100 * np.expm1(mean_log + variance / 2),
}, index=excl.index)
truncation.index.name = "ticker"
truncation.round(4).to_csv(config.TABLES / "truncation_identity.csv")
print("-> output/tables/truncation_identity.csv\n")

print("Equation (5.1), worked examples. The geometric column is the paper's "
      "printed figure;\nthe arithmetic column is the expected compounded simple "
      "return and is what the\ntext should quote if it calls the quantity an "
      "expected return:")
print(truncation.loc[["SCHW", "GD", "AMD"]].round(2).to_string())
