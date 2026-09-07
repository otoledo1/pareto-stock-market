"""Step 5. Cross-stock clustering of the excluded dates.

Table 9. Each stock's ten best days are compared against a null in which every
stock draws ten dates uniformly at random, without replacement, from the same
window. Heavy clustering on shared dates violates the independence assumption
behind the HC3 standard errors of Table 5.

Writes: output/tables/date_clustering.csv, excluded_dates.csv
"""

import numpy as np
import pandas as pd
from common import (config, drop_outlier, load_panel, percentile_of, summarize, write)

panel = load_panel()
tickers = [t for t in panel.columns if t != config.OUTLIER]
sessions = len(panel)

slots = []
for ticker in tickers:
    for date in panel[ticker].nlargest(config.K).index:
        slots.append({"ticker": ticker, "date": date.date()})
slots = pd.DataFrame(slots)
counts = slots.date.value_counts()

observed = {
    "distinct_dates": counts.size,
    "max_stocks_sharing_a_date": int(counts.iloc[0]),
    "shared_slot_percent": 100 * counts[counts > 1].sum() / len(slots),
}
print(f"{len(slots)} (stock, day) slots across {len(tickers)} stocks")
print(f"Most shared: {counts.index[0]} in {counts.iloc[0]} lists")
print(counts.head(5).to_string(), "\n")

rng = np.random.default_rng(config.SEED)
draws = {key: [] for key in observed}
for _ in range(config.N_OVERLAP_REPS):
    picks = np.concatenate([rng.choice(sessions, config.K, replace=False)
                            for _ in tickers])
    tally = np.bincount(picks, minlength=sessions)
    draws["distinct_dates"].append((tally > 0).sum())
    draws["max_stocks_sharing_a_date"].append(tally.max())
    draws["shared_slot_percent"].append(100 * tally[tally > 1].sum() / len(slots))

rows = []
for key, value in observed.items():
    band = summarize(draws[key])
    rows.append({"quantity": key, "actual": value, **band,
                 "pct_99_9": np.percentile(draws[key], 99.9),
                 "actual_percentile": percentile_of(draws[key], value)})
write(pd.DataFrame(rows).set_index("quantity").round(3), "date_clustering")
slots.to_csv(config.TABLES / "excluded_dates.csv", index=False)
print(f"-> output/tables/excluded_dates.csv")
