from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import yfinance as yf


DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]


def normalize_tickers(tickers: list[str]) -> list[str]:
    normalized = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not normalized:
        raise ValueError("At least one ticker is required.")
    return normalized


def make_output_path(
    output_dir: Path,
    tickers: list[str],
    start: str,
    end: str | None,
    interval: str,
) -> Path:
    end_label = end or date.today().isoformat()
    ticker_label = "_".join(tickers)
    filename = f"yfinance_{ticker_label}_{start}_{end_label}_{interval}.csv"
    return output_dir / filename


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    output_dir: Path = Path("data/raw"),
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
        threads=True,
    )

    if prices.empty:
        joined = ", ".join(tickers)
        raise RuntimeError(f"No data returned for: {joined}")

    output_path = make_output_path(output_dir, tickers, start, end, interval)
    prices.to_csv(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download market prices with yfinance.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--no-auto-adjust", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = download_prices(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=not args.no_auto_adjust,
        output_dir=args.output_dir,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
