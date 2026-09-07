"""Step 1. Retrieve adjusted closes and apply the four-step sample selection.

This reproduces the original data pull and checks the result against the
archived panel. It is provided for independent replication only: Yahoo Finance
revises adjusted price history over time, so a fresh run may not reproduce the
archived panel exactly. The archived export in data/raw is the object of record
for every number in the paper, and no later step reads this script's output.

Writes: data/raw/prices_refetched.csv, output/tables/refetch_comparison.csv
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

candidates = config.CANDIDATES.read_text().split()
print(f"Downloading {len(candidates)} candidates, "
      f"{config.START_DATE} to {config.END_DATE}")

raw = yf.download(candidates, start=config.START_DATE, end=config.END_DATE,
                  auto_adjust=True, progress=False)
closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
closes = closes.ffill().dropna(axis=1, how="all")

missing = [t for t in candidates if t not in closes.columns]
print(f"Retrieved {closes.shape[1]} tickers over {len(closes)} sessions")
if missing:
    print(f"No data returned for: {', '.join(missing)}")

# Measure the period return from each column's first and last valid observation,
# so a ticker that starts late is still evaluated over the days it did trade.
first_valid = closes.apply(lambda col: col.dropna().iloc[0])
last_valid = closes.apply(lambda col: col.dropna().iloc[-1])
period_return = (last_valid - first_valid) / first_valid * 100

net_positive = [t for t in candidates if t in closes.columns and period_return[t] > 0]
print(f"Net-positive over the window: {len(net_positive)} of {closes.shape[1]}")

selected = net_positive[:config.TOP_N]
print(f"Retained after top-{config.TOP_N} truncation: {len(selected)}")
if len(selected) < config.TOP_N:
    print(f"WARNING: fewer than {config.TOP_N} candidates passed the filter")

prices = closes[selected]
prices.index.name = "Date"
out = config.RAW / "prices_refetched.csv"
prices.to_csv(out)
print(f"-> {out.relative_to(config.ROOT)}\n")

archived = list(pd.read_csv(config.PANEL, index_col=0, nrows=0).columns)
added = [t for t in selected if t not in archived]
dropped = [t for t in archived if t not in selected]

print(f"Against the archived sample of {len(archived)} tickers:")
print(f"  in common {len(set(selected) & set(archived))}, "
      f"newly selected {len(added)}, no longer selected {len(dropped)}")
if added:
    print(f"  newly selected: {', '.join(added)}")
if dropped:
    print(f"  no longer selected: {', '.join(dropped)}")
if not added and not dropped:
    print("  the re-pull reproduces the archived sample exactly")

comparison = pd.DataFrame({
    "period_return_pct": period_return.reindex(sorted(set(selected) | set(archived))),
    "in_refetch": [t in selected for t in sorted(set(selected) | set(archived))],
    "in_archived": [t in archived for t in sorted(set(selected) | set(archived))],
})
comparison.index.name = "ticker"
comparison.round(4).to_csv(config.TABLES / "refetch_comparison.csv")
print(f"-> output/tables/refetch_comparison.csv")
