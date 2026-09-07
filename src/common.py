"""Functions shared across the analysis scripts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def load_panel():
    """Return the 249x100 daily return panel, dates ascending."""
    panel = pd.read_csv(config.PANEL, index_col=0, parse_dates=True)
    return panel.sort_index()


def load_sectors():
    """Return a ticker -> GICS sector mapping."""
    table = pd.read_csv(config.SECTORS)
    return dict(zip(table.ticker, table.sector))


def annualized_volatility(returns):
    return returns.std(ddof=1) * np.sqrt(config.TRADING_DAYS) * 100


def compound(returns):
    return np.prod(1 + returns) - 1


def excluded_indices(returns, k, tail="best"):
    """Positions of the k most extreme days: best, worst, or both tails."""
    order = np.argsort(returns)
    if tail == "best":
        return order[-k:]
    if tail == "worst":
        return order[:k]
    if tail == "both":
        return np.concatenate([order[-k:], order[:k]])
    raise ValueError(f"unknown tail: {tail}")


def miss_return(returns, k, tail="best"):
    """Compound return with the k most extreme days earning zero."""
    return compound(np.delete(returns, excluded_indices(returns, k, tail)))


def stock_statistics(panel, k=None):
    """Per-stock buy-and-hold, miss-top-k, gap and volatilities."""
    k = config.K if k is None else k
    rows = {}
    for ticker in panel.columns:
        returns = panel[ticker].values
        kept = np.delete(returns, excluded_indices(returns, k))
        buy_hold = compound(returns)
        missed = compound(kept)
        rows[ticker] = {
            "buy_hold": buy_hold * 100,
            "miss_top_k": missed * 100,
            "gap": (buy_hold - missed) * 100,
            "volatility": annualized_volatility(returns),
            "volatility_leaveout": annualized_volatility(kept),
        }
    return pd.DataFrame(rows).T


def drop_outlier(frame):
    return frame.drop(config.OUTLIER)


def fit(y, x):
    """OLS of y on x with an intercept."""
    return sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit()


def r_squared(x, y):
    return fit(y, x).rsquared


def summarize(values):
    """Median and 95% range of a Monte Carlo distribution."""
    values = np.asarray(values, float)
    return {
        "median": np.median(values),
        "lower_95": np.percentile(values, 2.5),
        "upper_95": np.percentile(values, 97.5),
    }


def percentile_of(values, observed):
    return 100 * (np.asarray(values, float) < observed).mean()


def write(frame, name):
    """Save a table to output/tables and echo it."""
    path = config.TABLES / f"{name}.csv"
    frame.to_csv(path)
    print(frame.to_string())
    print(f"\n-> {path.relative_to(config.ROOT)}\n")
