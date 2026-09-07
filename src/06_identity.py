"""Step 6. The exact gap identity and the symmetric timing tests.

Section 5.5 and Table 12. The gap decomposes exactly as
gap = 100 * (1 + R_buy_hold) * (1 - 1/A), where A is the compounded gross
return on a stock's own ten excluded days. L = 1 - 1/A is the bounded share of
terminal wealth those days account for. The symmetric tests remove the ten
worst days, and both tails, for comparison.

Writes: output/tables/identity.csv, symmetric_exclusion.csv
"""

import numpy as np
import pandas as pd
from common import (compound, config, drop_outlier, load_panel, miss_return,
                    stock_statistics, write)

panel = load_panel()
results = stock_statistics(panel)

A = (1 + results.buy_hold / 100) / (1 + results.miss_top_k / 100)
identity = pd.DataFrame({
    "gross_return_on_top_days": A,
    "stand_alone_return_pct": 100 * (A - 1),
    "wealth_share_L_pct": 100 * (1 - 1 / A),
    "gap_reported": results.gap,
    "gap_from_identity": 100 * (1 + results.buy_hold / 100) * (1 - 1 / A),
})
identity["abs_error"] = (identity.gap_reported - identity.gap_from_identity).abs()
print(f"Maximum absolute identity error across {len(identity)} stocks: "
      f"{identity.abs_error.max():.2e} pp\n")

excl = drop_outlier(identity)
print(f"Stand-alone return excl. {config.OUTLIER}: "
      f"{excl.stand_alone_return_pct.min():.1f}% ({excl.stand_alone_return_pct.idxmin()}) to "
      f"{excl.stand_alone_return_pct.max():.1f}% ({excl.stand_alone_return_pct.idxmax()}), "
      f"median {excl.stand_alone_return_pct.median():.1f}%")
print(f"{config.OUTLIER}: {identity.stand_alone_return_pct[config.OUTLIER]:.1f}%\n")

excl_results = drop_outlier(results)
L = drop_outlier(identity).wealth_share_L_pct
print(f"corr(L, volatility) {np.corrcoef(L, excl_results.volatility)[0, 1]:.3f}   "
      f"corr(L, buy-and-hold) {np.corrcoef(L, excl_results.buy_hold)[0, 1]:.3f}   "
      f"corr(gap, buy-and-hold) "
      f"{np.corrcoef(excl_results.gap, excl_results.buy_hold)[0, 1]:.3f}\n")
write(identity.sort_values("gap_reported", ascending=False).round(6), "identity")

rows = {}
for ticker in panel.columns:
    returns = panel[ticker].values
    buy_hold = compound(returns)
    rows[ticker] = {
        "miss_best": (buy_hold - miss_return(returns, config.K, "best")) * 100,
        "miss_worst": (buy_hold - miss_return(returns, config.K, "worst")) * 100,
        "miss_both": (buy_hold - miss_return(returns, config.K, "both")) * 100,
    }
symmetric = drop_outlier(pd.DataFrame(rows).T)
write(pd.DataFrame({"mean": symmetric.mean(), "median": symmetric.median()}).round(2),
      "symmetric_exclusion")
