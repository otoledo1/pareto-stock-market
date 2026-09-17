"""Step 11. Exact per-stock conditional test of top-10 concentration.

Table 11 and the S columns of Table S2. For stock i, let z_it be the daily log
returns standardized by that stock's own sample mean and sample standard
deviation, and let S_i be the sum of the ten largest z_it.

Under an i.i.d. Gaussian model the standardized vector is ancillary: its
distribution depends on neither mu_i nor sigma_i, so one reference distribution
serves every stock and no per-stock calibration, drift matching or rejection
sampling is required. The return-level defect that biases the drift-calibrated
null of step 7 therefore cannot arise here. By the same argument the winner
conditioning of section 3, which is a selection on the sample mean, cannot bias
S_i under the null.

The reference law is exact; its evaluation is not. It is obtained from 500,000
simulated standardized Gaussian vectors, in chunks to keep peak memory near
100 MB. What the test avoids is calibration and per-stock simulation, not
simulation itself.

p-values follow Phipson and Smyth (2010): p = (b + 1) / (B + 1), where b counts
reference statistics at least as large as the observation. Zero exceedances
give 1 / 500001, a floor rather than a certified tail bound.

Individual p-values are valid one stock at a time. Because the stocks share
calendar dates, the count of rejections is a descriptive summary and not a
valid joint test -- so the summary also reports Bonferroni counts, which are
valid under arbitrary dependence, and the cross-sectional dispersion of S_i,
which step 13 benchmarks against a null that carries the sample's correlation
structure.

Writes: output/tables/exact_conditional_test.csv     (one row per stock)
        output/tables/exact_conditional_summary.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from common import config, load_panel, monte_carlo_p, top_k_standardized, write

N_REFERENCE = 500_000
CHUNK = 50_000

panel = load_panel().drop(columns=[config.OUTLIER])
log_returns = np.log1p(panel.to_numpy())
sessions, n_stocks = log_returns.shape
print(f"panel: {sessions} sessions x {n_stocks} stocks (excl. {config.OUTLIER})")

S = top_k_standardized(log_returns)

rng = np.random.default_rng(config.SEED)
pieces, drawn = [], 0
while drawn < N_REFERENCE:
    size = min(CHUNK, N_REFERENCE - drawn)
    draw = rng.standard_normal((size, sessions))
    draw = (draw - draw.mean(axis=1, keepdims=True)) / draw.std(axis=1, ddof=1, keepdims=True)
    pieces.append(np.sort(draw, axis=1)[:, -config.K:].sum(axis=1))
    drawn += size
reference = np.concatenate(pieces)
reference.sort()

p_value = monte_carlo_p(reference, S)

# The paper's kurtosis figures use the bias-corrected estimator. The biased one
# gives 3.01 and 87 of 99, so the convention is worth recording alongside.
kurtosis = stats.kurtosis(log_returns, axis=0, bias=False)
kurtosis_biased = stats.kurtosis(log_returns, axis=0, bias=True)
skew = stats.skew(log_returns, axis=0, bias=False)

per_stock = pd.DataFrame({
    "S": S,
    "p_one_sided": p_value,
    "excess_kurtosis": kurtosis,
    "skew": skew,
    "worst_day_pct": 100 * panel.to_numpy().min(axis=0),
}, index=panel.columns).sort_values("S", ascending=False)
per_stock.index.name = "ticker"
per_stock.round(6).to_csv(config.TABLES / "exact_conditional_test.csv")
print(f"-> output/tables/exact_conditional_test.csv\n")

lower, upper = np.percentile(reference, [2.5, 97.5])
bonferroni_05 = int((p_value < 0.05 / n_stocks).sum())
bonferroni_01 = int((p_value < 0.01 / n_stocks).sum())

summary = pd.DataFrame([{
    "null_mean": reference.mean(),
    "null_sd": reference.std(ddof=1),
    "null_p2_5": lower,
    "null_p97_5": upper,
    "null_p99_9": np.percentile(reference, 99.9),
    "actual_median_S": float(np.median(S)),
    "actual_min_S": float(S.min()),
    "actual_max_S": float(S.max()),
    "actual_sd_of_S": float(S.std(ddof=1)),
    "n_stocks": n_stocks,
    "n_above_p97_5": int((S > upper).sum()),
    "n_below_p2_5": int((S < lower).sum()),
    "n_p_below_0_001": int((p_value < 0.001).sum()),
    "n_bonferroni_5pct": bonferroni_05,
    "n_bonferroni_1pct": bonferroni_01,
    "bonferroni_5pct_threshold": 0.05 / n_stocks,
    "median_excess_kurtosis": float(np.median(kurtosis)),
    "n_excess_kurtosis_above_1": int((kurtosis > 1).sum()),
    "median_excess_kurtosis_biased": float(np.median(kurtosis_biased)),
    "n_excess_kurtosis_above_1_biased": int((kurtosis_biased > 1).sum()),
}])
write(summary.T.round(6), "exact_conditional_summary")

print(f"Bonferroni: {bonferroni_05} of {n_stocks} stocks reject at a family-wise 5% "
      f"level (p < {0.05 / n_stocks:.3e}), {bonferroni_01} at 1%. Valid under "
      f"arbitrary cross-stock dependence.\n")

print(f"Cross-sectional sd of S: {S.std(ddof=1):.4f} "
      f"(step 13 benchmarks this against a correlated Gaussian null)\n")

print(f"Stocks below the 2.5th percentile ({lower:.4f}) -- all downside-fat-tailed, "
      f"not thin-tailed:")
print(per_stock[per_stock.S < lower]
      [["S", "worst_day_pct", "excess_kurtosis", "skew"]].round(2).to_string())
