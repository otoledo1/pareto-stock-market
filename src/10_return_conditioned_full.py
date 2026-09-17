"""Step 10. The return-conditioned Gaussian benchmark, under four conventions.

Table 10. Each simulated log-return path is de-meaned and rescaled so its sum
equals the stock's realized annual log return exactly and its sample standard
deviation equals a target exactly. The loop records all four headline
statistics, so Table 10 reports the strict null for every one of them rather
than for the flip rate alone.

Two conventions in that construction are not forced by the design, and the
second-round referees disagreed about whether they matter. Both are now run:

  scale       "simple"  target sd is s_i = sigma_i / sqrt(252), where sigma_i
                        is annualized volatility computed from simple returns.
                        This is the published convention.
              "log"     target sd is the stock's own daily log-return standard
                        deviation. The two differ by a factor of 0.95 to 1.10
                        across the cross-section (median 0.999), with SNPS at
                        the top end (62.11% against a printed 56.46%).

  regressor   "fixed"       the full-volatility R^2 row regresses on the
                            observed sigma_i, held fixed across replications.
                            This is the published convention.
              "recomputed"  the regressor is re-estimated from each simulated
                            path, matching what the leave-out row already does.

The published row is (simple, fixed) and its draw order is unchanged, so it
reproduces the printed figures exactly. The headline findings -- the observed
median gap above every replication and the leave-out R^2 below every
replication -- hold under all four. The full-volatility R^2 does not: it falls
below the null under three conventions and lands inside it, near the fifth
percentile, under (log, recomputed). Report it as specification-dependent.

Each replication draws one (249, 99) standard-normal block, not 99 separate
per-stock vectors. The two differ in draw order and shift the conditioned flip
p-value from 0.174 to about 0.155.

Writes: output/tables/return_conditioned.csv        (one row per statistic per
                                                     convention)
        output/tables/return_conditioned_draws.csv  (one row per replication)
"""

import numpy as np
import pandas as pd
from common import (config, cross_section, drop_outlier, load_panel,
                    percentile_of, r_squared, stock_statistics, summarize, write)

SCALES = ("simple", "log")
REGRESSORS = ("fixed", "recomputed")
STATISTICS = ("median_gap", "r_squared", "leaveout_r_squared", "flip_rate")

panel = load_panel()
sessions = len(panel)
excl_panel = panel.drop(columns=[config.OUTLIER])
excl = drop_outlier(stock_statistics(panel))
n_stocks = len(excl)

log_returns = np.log1p(excl_panel.to_numpy())
annual_log_return = log_returns.sum(axis=0)          # matched exactly, per path
volatility = excl.volatility.values                   # annualized %, observed

targets = {
    "simple": volatility / 100 / np.sqrt(config.TRADING_DAYS),
    "log": log_returns.std(axis=0, ddof=1),
}
ratio = targets["log"] * np.sqrt(config.TRADING_DAYS) * 100 / volatility
print(f"annualized log sd / printed simple sd: min {ratio.min():.3f}, "
      f"median {np.median(ratio):.3f}, max {ratio.max():.3f} "
      f"({excl.index[ratio.argmax()]})")

observed = {
    "median_gap": float(np.median(excl.gap.values)),
    "r_squared": r_squared(volatility, excl.gap.values),
    "leaveout_r_squared": r_squared(excl.volatility_leaveout.values, excl.gap.values),
    "flip_rate": 100 * float((excl.miss_top_k.values < 0).mean()),
}
print("actual:", {k: round(v, 4) for k, v in observed.items()}, "\n")


def replicate(scale):
    """One pass of N_BENCHMARK_REPS, recording both regressor conventions."""
    rng = np.random.default_rng(config.SEED)
    target = targets[scale]
    rows = []
    for index in range(config.N_BENCHMARK_REPS):
        z = rng.standard_normal((sessions, n_stocks))
        z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)
        simulated = np.expm1(annual_log_return / sessions + target * z)
        stats = cross_section(simulated)
        rows.append({
            "median_gap": np.median(stats["gap"]),
            "r_squared_fixed": r_squared(volatility, stats["gap"]),
            "r_squared_recomputed": r_squared(stats["volatility"], stats["gap"]),
            "leaveout_r_squared": r_squared(stats["volatility_leaveout"], stats["gap"]),
            "flip_rate": 100 * (stats["miss_top_k"] < 0).mean(),
        })
        if (index + 1) % 500 == 0:
            print(f"  {scale}: {index + 1}/{config.N_BENCHMARK_REPS}")
    return pd.DataFrame(rows)


draws = {scale: replicate(scale) for scale in SCALES}

rows, wide = [], {}
for scale in SCALES:
    for regressor in REGRESSORS:
        frame = draws[scale]
        columns = {"median_gap": "median_gap",
                   "r_squared": f"r_squared_{regressor}",
                   "leaveout_r_squared": "leaveout_r_squared",
                   "flip_rate": "flip_rate"}
        for statistic in STATISTICS:
            values = frame[columns[statistic]].to_numpy()
            wide[f"{scale}_{regressor}_{statistic}"] = values
            rows.append({
                "quantity": statistic, "scale": scale, "regressor": regressor,
                "actual": observed[statistic], **summarize(values),
                "actual_percentile": percentile_of(values, observed[statistic]),
                "published": scale == "simple" and regressor == "fixed",
            })

table = pd.DataFrame(rows).set_index(["quantity", "scale", "regressor"]).round(4)
write(table, "return_conditioned")
pd.DataFrame(wide).to_csv(config.TABLES / "return_conditioned_draws.csv", index=False)
print("-> output/tables/return_conditioned_draws.csv\n")

print("Percentile of the observed value, by convention:")
print(table.loc[list(STATISTICS)][["actual_percentile"]].to_string())
