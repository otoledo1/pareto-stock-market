"""Step 3. Buy-and-hold against miss-top-10, per stock.

Builds the core results table used by every later script, and reports the
summary statistics of Tables 1 and 2 and the flip-rate quartiles of Table 4.

Writes: output/tables/stock_results.csv, summary_statistics.csv, flip_by_quartile.csv
"""

import numpy as np
import pandas as pd
from common import (config, drop_outlier, load_panel, stock_statistics, write)

panel = load_panel()
results = stock_statistics(panel)
results["buy_hold_value"] = config.INITIAL_CAPITAL * (1 + results.buy_hold / 100)
results["miss_top_k_value"] = config.INITIAL_CAPITAL * (1 + results.miss_top_k / 100)

print(f"Median gap: {drop_outlier(results).gap.median():.2f} pp excl. {config.OUTLIER}, "
      f"{results.gap.median():.2f} pp incl.")
print(f"Flipped to negative: {int((results.miss_top_k < 0).sum())} of {len(results)}\n")

write(results.sort_values("gap", ascending=False).round(4), "stock_results")

rows = []
for label, frame in [("excl_MU", drop_outlier(results)), ("incl_MU", results)]:
    for column in ["buy_hold", "miss_top_k", "gap", "volatility"]:
        series = frame[column]
        rows.append({"sample": label, "variable": column, "n": len(series),
                     "mean": series.mean(), "sd": series.std(ddof=1),
                     "median": series.median(), "min": series.min(), "max": series.max()})
write(pd.DataFrame(rows).round(2).set_index(["sample", "variable"]), "summary_statistics")

quartile = pd.qcut(results.buy_hold, 4, labels=["Q1", "Q2", "Q3", "Q4"])
rows = []
for label in ["Q1", "Q2", "Q3", "Q4"]:
    group = results[quartile == label]
    rows.append({"quartile": label, "n": len(group),
                 "bh_min": group.buy_hold.min(), "bh_max": group.buy_hold.max(),
                 "flip_rate": 100 * (group.miss_top_k < 0).mean()})
write(pd.DataFrame(rows).round(2).set_index("quartile"), "flip_by_quartile")
