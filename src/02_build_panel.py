"""Step 2. Normalize the archived export into the analysis panel.

The archived spreadsheet export interleaves price and return columns, with
returns carrying a '.1' suffix. This extracts the return columns, strips the
suffix, and writes a clean panel of daily simple returns.

Writes: data/panel_returns.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

export = pd.read_csv(config.SHEETS_EXPORT, index_col=0, parse_dates=True)
export = export.apply(pd.to_numeric, errors="coerce")

panel = export[[c for c in export.columns if c.endswith(".1")]]
panel.columns = panel.columns.str.removesuffix(".1")
panel = panel.dropna(how="all").sort_index()

assert panel.shape == (249, 100), f"expected 249x100, got {panel.shape}"
assert not panel.isna().any().any(), "panel contains missing values"

panel.to_csv(config.PANEL)
print(f"Panel: {panel.shape[0]} sessions x {panel.shape[1]} tickers")
print(f"Window: {panel.index[0].date()} to {panel.index[-1].date()}")
print(f"-> {config.PANEL.relative_to(config.ROOT)}")
