"""Step 12. Block bootstrap over calendar dates for the volatility-gap slope.

Table 9. Section 5.3 documents that the excluded dates cluster on a small
number of market-wide sessions, so the 99 observations are not independent and
neither conventional nor HC3 standard errors are valid. Resampling blocks of
trading days, keeping the whole cross-section together and recomputing the gap,
the volatility and the regression inside each resample, preserves both the
serial dependence within a stock and the shared-date dependence across stocks.

This holds the 99 stocks fixed and treats the year of trading days as the
random sample. Three implementation choices the first version left implicit,
all of which the second-round referees asked to see:

  scheme    "moving"      non-circular blocks, ceil(249 / l) of them, uniform
                          starts in [0, 249 - l], concatenated and truncated to
                          249 days. This is the published scheme.
            "circular"    starts uniform on [0, 249), wrapping at the end.
            "stationary"  geometric block lengths with mean l (Politis and
                          Romano), so no fixed block boundary.

  screen    False  every stock is kept, whatever its resampled annual return.
                   This is the published variant, and it is inconsistent with
                   section 3: in a resampled year only about 80 of the 99
                   stocks remain net-positive, so a fifth of each pseudo-sample
                   would not have entered the original sample.
            True   the section 3 winner screen is re-applied inside each
                   resample. The standard error is unchanged at 1.7 to 2.0, but
                   the interval's lower end rises to about 1.6 and the
                   bootstrap median to 3.2 to 3.6.

One generator is shared across block lengths 5, 10 and 21, drawn in that order,
within each (scheme, screen) pass. The published row is (moving, False) and
reproduces the printed figures exactly.

The bootstrap distribution is right-skewed, so the percentile intervals are not
bias-corrected; the bootstrap median is reported beside them. Resampling dates
also perturbs each stock's realized annual return, so these are
dependence-sensitive resampling ranges, not validated post-selection confidence
intervals.

Writes: output/tables/block_bootstrap.csv        (one row per variant)
        output/tables/block_bootstrap_draws.csv  (one column per variant)
"""

import numpy as np
import pandas as pd
from common import (config, cross_section, drop_outlier, load_panel, ols_slope,
                    stock_statistics, write)

BLOCK_LENGTHS = (5, 10, 21)
SCHEMES = ("moving", "circular", "stationary")
MIN_SURVIVORS = 10

panel = load_panel().drop(columns=[config.OUTLIER])
returns = panel.to_numpy()
sessions, n_stocks = returns.shape
point = ols_slope(drop_outlier(stock_statistics(load_panel())).volatility,
                  drop_outlier(stock_statistics(load_panel())).gap)
print(f"panel: {sessions} sessions x {n_stocks} stocks")
print(f"point slope: {point:.6f}\n")


def block_index(rng, length, scheme):
    """Row positions for one resampled year of trading days."""
    if scheme == "moving":
        starts = rng.integers(0, sessions - length + 1, int(np.ceil(sessions / length)))
        return np.concatenate([np.arange(s, s + length) for s in starts])[:sessions]
    if scheme == "circular":
        starts = rng.integers(0, sessions, int(np.ceil(sessions / length)))
        return np.concatenate([np.arange(s, s + length) % sessions
                               for s in starts])[:sessions]
    if scheme == "stationary":
        probability, index = 1.0 / length, []
        while len(index) < sessions:
            position = int(rng.integers(0, sessions))
            index.append(position)
            while len(index) < sessions and rng.random() > probability:
                position = (position + 1) % sessions
                index.append(position)
        return np.array(index[:sessions])
    raise ValueError(f"unknown scheme: {scheme}")


def resample_slope(index, screen):
    """Slope on one resampled year, optionally re-applying the winner screen."""
    stats = cross_section(returns[index])
    kept = stats["buy_hold"] > 0 if screen else np.ones(n_stocks, bool)
    if kept.sum() < MIN_SURVIVORS:
        return np.nan, int(kept.sum())
    return ols_slope(stats["volatility"][kept], stats["gap"][kept]), int(kept.sum())


rows, all_draws = [], {}
for scheme in SCHEMES:
    for screen in (False, True):
        rng = np.random.default_rng(config.SEED)
        for length in BLOCK_LENGTHS:
            slopes = np.empty(config.N_BENCHMARK_REPS)
            survivors = np.empty(config.N_BENCHMARK_REPS)
            for replication in range(config.N_BENCHMARK_REPS):
                slopes[replication], survivors[replication] = resample_slope(
                    block_index(rng, length, scheme), screen)
            valid = slopes[~np.isnan(slopes)]
            label = f"{scheme}_{'screened' if screen else 'unscreened'}_{length}"
            all_draws[label] = slopes
            rows.append({
                "scheme": scheme, "screen": screen, "block_length": length,
                "point_slope": point,
                "bootstrap_se": valid.std(ddof=1),
                "bootstrap_median": np.median(valid),
                "ci_2_5": np.percentile(valid, 2.5),
                "ci_97_5": np.percentile(valid, 97.5),
                "mean_stocks_kept": survivors.mean(),
                "published": scheme == "moving" and not screen,
            })
        print(f"  {scheme}, screen={screen}: done")

table = pd.DataFrame(rows).set_index(["scheme", "screen", "block_length"]).round(4)
write(table, "block_bootstrap")
pd.DataFrame(all_draws).to_csv(config.TABLES / "block_bootstrap_draws.csv", index=False)
print("-> output/tables/block_bootstrap_draws.csv\n")

spread = table.bootstrap_se
print(f"Standard error across all schemes and block lengths: "
      f"{spread.min():.2f} to {spread.max():.2f}; "
      f"conventional {0.264:.3f}, HC3 {0.693:.3f}")
