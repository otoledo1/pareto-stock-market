"""Step 13. A return-conditioned Gaussian null that carries the sample's
cross-stock correlation matrix.

Step 11's per-stock p-values are valid one stock at a time, but the count of
rejections across the cross-section is not a joint test, because the stocks
share calendar dates. Two things answer that, and this script supplies the
second:

  (a) Bonferroni, reported by step 11, which is valid under arbitrary
      dependence: 23 of 99 stocks reject at a family-wise 5% level.
  (b) A null that reproduces the sample's own cross-stock correlation.

Each replication draws a (249, 99) matrix from N(0, C), where C is the
correlation matrix of the panel's daily log returns (mean pairwise correlation
0.146), standardizes each column and maps it through the same return-
conditioned construction as step 10.

The result is that Gaussian correlation of this size barely moves the extreme
order statistics: every actual value still falls outside the null's range. That
is itself informative. The observed clustering documented in step 5 -- 45 of 99
stocks sharing 2026-04-08 -- is stronger than any Gaussian factor structure of
this correlation produces, which is a non-Gaussian feature of the data and the
reason (a), not (b), is the fully dependence-robust statement.

This also benchmarks the cross-sectional dispersion of S_i, which is the
direct evidence that concentration varies across names. It is the statistic
section 7 should rest on rather than the cross-sectional R^2, whose departure
step 10 shows to be specification-dependent.

Requires: output/tables/exact_conditional_summary.csv (run step 11 first)
Writes:   output/tables/correlated_null.csv
          output/tables/correlated_null_draws.csv
"""

import numpy as np
import pandas as pd
from common import (config, cross_section, drop_outlier, load_panel,
                    percentile_of, r_squared, stock_statistics,
                    top_k_standardized, summarize, write)

panel = load_panel()
sessions = len(panel)
excl_panel = panel.drop(columns=[config.OUTLIER])
excl = drop_outlier(stock_statistics(panel))
n_stocks = len(excl)

log_returns = np.log1p(excl_panel.to_numpy())
annual_log_return = log_returns.sum(axis=0)
volatility = excl.volatility.values
daily_sd = volatility / 100 / np.sqrt(config.TRADING_DAYS)

correlation = np.corrcoef(log_returns.T)
upper = np.triu_indices(n_stocks, 1)
print(f"mean pairwise correlation of daily log returns: "
      f"{correlation[upper].mean():.4f}")

try:
    factor = np.linalg.cholesky(correlation)
except np.linalg.LinAlgError:                 # near-singular, fall back
    values, vectors = np.linalg.eigh(correlation)
    factor = vectors @ np.diag(np.sqrt(np.clip(values, 1e-12, None)))

summary_path = config.TABLES / "exact_conditional_summary.csv"
reference = pd.read_csv(summary_path, index_col=0).iloc[:, 0]
threshold_975 = float(reference["null_p97_5"])
threshold_999 = float(reference["null_p99_9"])
print(f"reference thresholds from step 11: 97.5th {threshold_975:.4f}, "
      f"99.9th {threshold_999:.4f}")

S_actual = top_k_standardized(log_returns)
observed = {
    "n_above_p97_5": float((S_actual > threshold_975).sum()),
    "n_above_p99_9": float((S_actual > threshold_999).sum()),
    "sd_of_S": float(S_actual.std(ddof=1)),
    "median_gap": float(np.median(excl.gap.values)),
    "r_squared": r_squared(volatility, excl.gap.values),
    "leaveout_r_squared": r_squared(excl.volatility_leaveout.values, excl.gap.values),
}
print("actual:", {k: round(v, 4) for k, v in observed.items()}, "\n")

rng = np.random.default_rng(config.SEED)
rows = []
for replication in range(config.N_BENCHMARK_REPS):
    draw = rng.standard_normal((sessions, n_stocks)) @ factor.T
    z = (draw - draw.mean(axis=0)) / draw.std(axis=0, ddof=1)
    simulated_log = annual_log_return / sessions + daily_sd * z
    stats = cross_section(np.expm1(simulated_log))
    S = top_k_standardized(simulated_log)
    rows.append({
        "n_above_p97_5": (S > threshold_975).sum(),
        "n_above_p99_9": (S > threshold_999).sum(),
        "sd_of_S": S.std(ddof=1),
        "median_gap": np.median(stats["gap"]),
        "r_squared": r_squared(volatility, stats["gap"]),
        "leaveout_r_squared": r_squared(stats["volatility_leaveout"], stats["gap"]),
    })
    if (replication + 1) % 500 == 0:
        print(f"  {replication + 1}/{config.N_BENCHMARK_REPS}")

draws = pd.DataFrame(rows)
draws.to_csv(config.TABLES / "correlated_null_draws.csv", index=False)

table = pd.DataFrame([
    {"quantity": key, "actual": observed[key], **summarize(draws[key].to_numpy()),
     "actual_percentile": percentile_of(draws[key].to_numpy(), observed[key])}
    for key in observed
]).set_index("quantity").round(4)
write(table, "correlated_null")
print("-> output/tables/correlated_null_draws.csv")
