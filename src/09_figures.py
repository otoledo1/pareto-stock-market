"""Step 9. The two manuscript figures.

Figure 1 plots annualized volatility against the performance gap with an OLS
trendline. Figure 2 plots buy-and-hold against miss-top-10 returns with the
45-degree equal-return reference. Both exclude the outlier, whose coordinates
fall far outside the range needed to display the rest legibly.

Writes: output/figures/volatility_vs_gap.png, buy_hold_vs_miss.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import config, drop_outlier, fit, load_panel, stock_statistics

excl = drop_outlier(stock_statistics(load_panel()))
LABELLED = ["AMD", "LRCX", "MRVL", "AMAT", "ORCL", "GOOGL", "DUK", "SO"]

model = fit(excl.gap, excl.volatility)
tercile = excl.buy_hold.rank(pct=True)
colors = np.where(tercile > 2 / 3, "#2e8b57",
                  np.where(tercile > 1 / 3, "#e0a800", "#c0392b"))

figure, axes = plt.subplots(figsize=(8, 5.5))
axes.scatter(excl.volatility, excl.gap, c=colors, s=26, zorder=3)
grid = np.linspace(excl.volatility.min(), excl.volatility.max(), 100)
axes.plot(grid, model.params[0] + model.params[1] * grid, "--", color="#333", lw=1.2,
          label=f"OLS: gap = {model.params[0]:.1f} + {model.params[1]:.2f} x vol "
                f"($R^2$={model.rsquared:.2f})")
for ticker in LABELLED:
    axes.annotate(ticker, (excl.volatility[ticker], excl.gap[ticker]),
                  fontsize=7, xytext=(4, 3), textcoords="offset points")
axes.set_xlabel("Annualized volatility (%)")
axes.set_ylabel("Performance gap (pp)")
axes.set_title(f"Annualized volatility vs. performance gap (N={len(excl)}, "
               f"{config.OUTLIER} excluded)", fontsize=11)
axes.legend(fontsize=8, frameon=True)
axes.grid(alpha=0.25, zorder=0)
figure.tight_layout()
figure.savefig(config.FIGURES / "volatility_vs_gap.png", dpi=200)

trend = fit(excl.miss_top_k, excl.buy_hold)
figure, axes = plt.subplots(figsize=(8, 5.5))
axes.scatter(excl.buy_hold, excl.miss_top_k, s=26, color="#3f7fbf", zorder=3)
limits = np.array([excl.buy_hold.min(), excl.buy_hold.max()])
axes.plot(limits, limits, "--", color="#c0392b", lw=1.2, label="45$^\\circ$ equal return")
axes.plot(limits, trend.params[0] + trend.params[1] * limits, "--", color="#4b0082",
          lw=1.2, label=f"OLS trend ($R^2$={trend.rsquared:.2f})")
axes.axhline(0, color="#888", lw=0.8, zorder=1)
axes.set_xlabel("Buy-and-hold return (%)")
axes.set_ylabel("Return after missing top 10 days (%)")
axes.set_title(f"Buy-and-hold vs. miss-top-10 return (N={len(excl)}, "
               f"{config.OUTLIER} excluded)", fontsize=11)
axes.legend(fontsize=8, frameon=True)
axes.grid(alpha=0.25, zorder=0)
figure.tight_layout()
figure.savefig(config.FIGURES / "buy_hold_vs_miss.png", dpi=200)

print(f"-> output/figures/volatility_vs_gap.png")
print(f"-> output/figures/buy_hold_vs_miss.png")
