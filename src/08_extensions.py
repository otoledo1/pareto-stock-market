"""Step 8. Rolling windows, exclusion thresholds and sector heterogeneity.

Tables 14, 15, 16 and 17. All three re-slice or re-parameterize the same panel,
so every extension stays consistent with the headline results.

Writes: output/tables/rolling_windows.csv, thresholds.csv, sectors.csv,
        sector_information_technology.csv
"""

import numpy as np
import pandas as pd
from common import (compound, config, drop_outlier, fit, load_panel, load_sectors,
                    miss_return, stock_statistics, write)

panel = load_panel()
results = stock_statistics(panel)
excl = drop_outlier(results)

WINDOW, STEP, N_WINDOWS = 126, 21, 6
rows = []
for index in range(N_WINDOWS):
    start = index * STEP
    slice_ = panel.iloc[start:start + WINDOW]
    window_results = stock_statistics(slice_)
    model = fit(drop_outlier(window_results).gap, drop_outlier(window_results).volatility)
    rows.append({"window": index + 1,
                 "start": slice_.index[0].date(), "end": slice_.index[-1].date(),
                 "mean_gap": window_results.gap.mean(),
                 "mean_volatility": window_results.volatility.mean(),
                 "slope": model.params[1], "r_squared": model.rsquared})
print(f"Consecutive windows overlap by {WINDOW - STEP} of {WINDOW} sessions "
      f"({100 * (WINDOW - STEP) / WINDOW:.0f}%)\n")
write(pd.DataFrame(rows).set_index("window").round(3), "rolling_windows")

rows = []
for k in [1, 5, 10, 15, 20]:
    gaps = pd.Series({t: (compound(panel[t].values) - miss_return(panel[t].values, k)) * 100
                      for t in panel.columns})
    model = fit(drop_outlier(gaps), excl.volatility)
    rows.append({"k": k, "mean_gap": drop_outlier(gaps).mean(),
                 "median_gap": drop_outlier(gaps).median(),
                 "outlier_gap": gaps[config.OUTLIER],
                 "slope": model.params[1], "r_squared": model.rsquared})
write(pd.DataFrame(rows).set_index("k").round(4), "thresholds")

sectors = load_sectors()
results["sector"] = results.index.map(sectors)
assert results.sector.notna().all(), "unmapped tickers"

tested = sum(1 for _, g in results.groupby("sector") if len(g) >= 4)
rows = []
for sector, group in results.groupby("sector"):
    row = {"sector": sector, "n": len(group),
           "mean_volatility": group.volatility.mean(), "mean_gap": group.gap.mean()}
    if len(group) >= 4:
        model = fit(group.gap, group.volatility)
        row.update({"slope": model.params[1], "r_squared": model.rsquared,
                    "p_value": model.pvalues[1],
                    "p_bonferroni": min(1.0, model.pvalues[1] * tested)})
    rows.append(row)
print(f"Sectors with N >= 4, used for the Bonferroni correction: {tested}\n")
write(pd.DataFrame(rows).set_index("sector").sort_values("n", ascending=False).round(4),
      "sectors")

tech = results[results.sector == "Information Technology"]
rows = []
for label, frame in [("incl. MU", tech), ("excl. MU", drop_outlier(tech))]:
    model = fit(frame.gap, frame.volatility)
    rows.append({"specification": label, "n": len(frame), "beta": model.params[1],
                 "se": model.bse[1], "r_squared": model.rsquared})
write(pd.DataFrame(rows).set_index("specification").round(4),
      "sector_information_technology")
