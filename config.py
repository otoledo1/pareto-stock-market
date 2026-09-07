"""Shared configuration. All paths are relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
OUTPUT = ROOT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

for _d in (OUTPUT, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SHEETS_EXPORT = RAW / "sheets_export_daily_returns.csv"
CANDIDATES = DATA / "candidate_tickers.txt"
SECTORS = DATA / "gics_sectors.csv"
PANEL = DATA / "panel_returns.csv"

START_DATE = "2025-06-01"
END_DATE = "2026-06-01"
TOP_N = 100

SEED = 20260712
K = 10
TRADING_DAYS = 252
N_BENCHMARK_REPS = 2000
N_OVERLAP_REPS = 5000
MAX_DRAW_ATTEMPTS = 200
INITIAL_CAPITAL = 1_000_000
OUTLIER = "MU"
