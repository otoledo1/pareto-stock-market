"""Step 7. The Gaussian benchmark.

Tables 8, 10 and 11. Simulates a market with no fat tails, matched to each
stock's own volatility and annual return, then applies the identical
miss-top-10 procedure. This measures how much of the observed effect is
produced by order statistics alone.

Two nulls are run. The drift-calibrated null matches each stock's annual return
in expectation and rejects non-positive draws, mirroring the sample's
positive-return filter. The return-conditioned null rescales each path so its
annual log return and daily standard deviation match exactly.

Writes: output/tables/gaussian_benchmark.csv, benchmark_replications.csv
"""

import numpy as np
import pandas as pd
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
}

print(f"Simulating {config.N_BENCHMARK_REPS} cross-sections of {n_stocks} stocks "
      f"x {sessions} sessions, seed {config.SEED}")

rng = np.random.default_rng(config.SEED)
drift_null = {"median_gap": [], "r_squared": [], "flip_rate": [], "leaveout_r_squared": []}

for _ in range(config.N_BENCHMARK_REPS):
    gaps = np.empty(n_stocks)
    leaveout_vol = np.empty(n_stocks)
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
        flips += missed < 0
        leaveout_vol[i] = kept.std(ddof=1) * np.sqrt(config.TRADING_DAYS) * 100
    drift_null["median_gap"].append(np.median(gaps))
    drift_null["r_squared"].append(r_squared(excl.volatility.values, gaps))
    drift_null["flip_rate"].append(100 * flips / n_stocks)
    drift_null["leaveout_r_squared"].append(r_squared(leaveout_vol, gaps))

rng = np.random.default_rng(config.SEED)
conditioned_flip = []
for _ in range(config.N_BENCHMARK_REPS):
    flips = 0
    for i in range(n_stocks):
        z = rng.standard_normal(sessions)
        z = (z - z.mean()) / z.std(ddof=1)
        returns = np.expm1(drift[i] + daily_sd[i] * z)
        kept = np.delete(returns, np.argsort(returns)[-config.K:])
        flips += (np.prod(1 + kept) - 1) < 0
    conditioned_flip.append(100 * flips / n_stocks)

rows = []
for key in ["median_gap", "r_squared", "flip_rate", "leaveout_r_squared"]:
    rows.append({"quantity": key, "null": "drift-calibrated", "actual": observed[key],
                 **summarize(drift_null[key]),
                 "actual_percentile": percentile_of(drift_null[key], observed[key])})
rows.append({"quantity": "flip_rate", "null": "return-conditioned",
             "actual": observed["flip_rate"], **summarize(conditioned_flip),
             "actual_percentile": percentile_of(conditioned_flip, observed["flip_rate"])})
write(pd.DataFrame(rows).set_index(["quantity", "null"]).round(4), "gaussian_benchmark")

replications = pd.DataFrame(drift_null)
replications["flip_rate_conditioned"] = conditioned_flip
replications.to_csv(config.TABLES / "benchmark_replications.csv", index=False)
print("-> output/tables/benchmark_replications.csv")
