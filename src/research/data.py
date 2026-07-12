from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


MARKET_ETFS = ["SPY", "QQQ", "TLT", "GLD", "SLV"]
MEGA_CAP_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "XOM", "UNH"]
EXTENDED_UNIVERSE = MARKET_ETFS + MEGA_CAP_STOCKS


def normalize_tickers(tickers: list[str]) -> list[str]:
    normalized = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not normalized:
        raise ValueError("At least one ticker is required.")
    return list(dict.fromkeys(normalized))


def make_yfinance_path(output_dir: Path, tickers: list[str], start: str, end: str | None, interval: str) -> Path:
    end_label = end or date.today().isoformat()
    ticker_label = "_".join(tickers)
    return output_dir / f"yfinance_{ticker_label}_{start}_{end_label}_{interval}.csv"


def download_yfinance_prices(
    tickers: list[str],
    *,
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
    output_dir: Path = Path("data/raw"),
    auto_adjust: bool = True,
    threads: bool = False,
) -> Path:
    tickers = normalize_tickers(tickers)
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
        threads=threads,
    )

    if prices.empty:
        raise RuntimeError(f"No data returned for: {', '.join(tickers)}")

    output_path = make_yfinance_path(output_dir, tickers, start, end, interval)
    prices.to_csv(output_path)
    return output_path


def load_yfinance_close(path: Path, tickers: list[str] | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    raw = raw[raw.index.notna()]
    raw.index.name = "Date"

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=1)
    else:
        close = raw[["Close"]]

    if tickers is not None:
        available = [ticker for ticker in tickers if ticker in close.columns]
        close = close[available]

    return close.sort_index().dropna(how="all")


def latest_raw_file(raw_dir: Path, prefix: str = "yfinance_") -> Path:
    candidates = sorted(raw_dir.glob(f"{prefix}*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No yfinance CSV files found in {raw_dir}")
    return candidates[0]

