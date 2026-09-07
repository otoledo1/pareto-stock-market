"""Step 4. Volatility-gap regressions and influence diagnostics.

Covers Table 5 (conventional and HC3 standard errors), Table 6 (Cook's
distance, leverage, studentized residuals, robust slopes) and Table 7 (the
leave-top-10-out regressor).

Writes: output/tables/regressions.csv, influence.csv, robustness.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import config, drop_outlier, fit, load_panel, stock_statistics, write

results = stock_statistics(load_panel())
excl = drop_outlier(results)


def specification(label, frame, regressor="volatility"):
    model = fit(frame.gap, frame[regressor])
    robust = model.get_robustcov_results("HC3")
    return {"specification": label, "n": len(frame),
            "alpha": model.params[0], "beta": model.params[1],
            "se_conventional": model.bse[1], "se_hc3": robust.bse[1],
            "t_conventional": model.tvalues[1], "t_hc3": robust.tvalues[1],
            "p_hc3": robust.pvalues[1], "r_squared": model.rsquared}


rows = [specification("Full sample (incl. MU)", results),
        specification("Excluding MU", excl),
        specification("Leave-top-10-out volatility", excl, "volatility_leaveout")]
for share_class in ["GOOG", "GOOGL"]:
    rows.append(specification(f"Excluding MU and {share_class}", excl.drop(share_class)))
write(pd.DataFrame(rows).set_index("specification").round(4), "regressions")

model = fit(excl.gap, excl.volatility)
influence = model.get_influence()
diagnostics = pd.DataFrame({
    "cooks_distance": influence.cooks_distance[0],
    "leverage": influence.hat_matrix_diag,
    "studentized_residual": influence.resid_studentized_external,
}, index=excl.index).sort_values("cooks_distance", ascending=False)
print(f"4/N reference threshold: {4 / len(excl):.4f}\n")
write(diagnostics.head(5).round(3), "influence")

rows = [{"estimator": "OLS, excl. MU", "beta": model.params[1], "r_squared": model.rsquared}]
without_amd = fit(excl.drop("AMD").gap, excl.drop("AMD").volatility)
rows.append({"estimator": "OLS, excl. MU and AMD",
             "beta": without_amd.params[1], "r_squared": without_amd.rsquared})
for norm, label in [(sm.robust.norms.HuberT(), "Huber"),
                    (sm.robust.norms.TukeyBiweight(), "Tukey biweight")]:
    robust = sm.RLM(np.asarray(excl.gap, float),
                    sm.add_constant(np.asarray(excl.volatility, float)), M=norm).fit()
    rows.append({"estimator": f"Robust ({label}), excl. MU",
                 "beta": robust.params[1], "r_squared": np.nan})
write(pd.DataFrame(rows).set_index("estimator").round(4), "robustness")
