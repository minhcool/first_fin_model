# Daily SPY Direction Prediction

This project tests whether daily market price features can predict the next-day direction of SPY and improve risk-adjusted returns versus simple buy-and-hold baselines.

The project is currently price-only. It does not use news, sentiment, earnings transcripts, fundamentals, macro releases, analyst ratings, order book data, or intraday data.

## Dataset

- Source: `yfinance`
- Frequency: daily bars, `interval="1d"`
- Start date: `2015-01-01`
- Main modeling file: `data/processed/extended_close_prices.csv`
- Target asset: `SPY`
- Prediction target: whether `SPY` has a positive next-day close-to-close return

The extended universe currently includes:

```text
SPY, QQQ, TLT, GLD, SLV,
AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA,
JPM, XOM, UNH
```

The ML notebooks mostly use daily close prices to build returns, momentum, volatility, correlation, drawdown, and trend features.

## Trading Setup

The main ML strategy is a daily long/cash SPY timing strategy:

- If the model predicts SPY up tomorrow: hold `100% SPY`
- If the model predicts SPY down tomorrow: hold `100% cash`
- No shorting
- No leverage
- No intraday trading
- Signals are evaluated out of sample with walk-forward splits

## Assumptions

- Transaction cost: `1.0` basis point per amount traded in the ML notebook
- This is equivalent to `0.01%` of traded notional
- A full switch from cash to SPY costs `0.01%`
- A full round trip from cash to SPY and back to cash costs about `0.02%`
- No taxes are modeled
- No bid-ask spread or slippage is modeled beyond the transaction-cost assumption
- Results use historical daily data and should not be treated as live trading performance

## Current Models

The walk-forward ML notebook tests:

- Logistic Regression
- Ridge Classifier
- Random Forest
- Gradient Boosting

The current best-performing model in `notebooks/07_walk_forward_ml.ipynb` is `logistic_regression`, but this should be treated as a research result, not proof that the model is robust.

## Strengths

- Uses out-of-sample walk-forward testing instead of training and testing on the same dates
- Compares ML models against buy-and-hold and simpler baselines
- Includes transaction costs when the position changes
- Tracks trading behavior with exposure and annual turnover
- Reports risk-adjusted metrics such as Sharpe, volatility, max drawdown, and Calmar ratio
- Keeps reusable backtest, data, feature, and strategy helpers in `src/`

## Limitations

- Uses only price-derived features, so it ignores news, fundamentals, macro conditions, and event risk
- Uses a relatively small set of assets and a limited set of ML models
- The strategy trades only SPY or cash, so it does not test broader portfolio allocation
- The backtest assumes daily close-to-close execution and simplified trading costs
- Hyperparameter tuning and model selection are still limited
- Results may be sensitive to the chosen date range, market regime, and transaction-cost assumption
- More robustness checks are needed before making any live-trading claim

## Project Layout

- `notebooks/` - exploratory notebooks and modeling experiments
- `src/data/` - reusable data download code
- `src/research/` - reusable research, feature, strategy, and backtest helpers
- `data/raw/` - downloaded source data from `yfinance` kept local by `.gitignore`
- `data/processed/` - cleaned/generated datasets kept local by `.gitignore`
- `reports/` - exported summary reports

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Download Market Data

Starter market data:

```powershell
python -m src.data.download_yfinance --tickers SPY QQQ TLT GLD --start 2015-01-01
```

Extended universe data is generated in:

```text
notebooks/04_extended_universe_data_pull.ipynb
```

## Notebooks

Open the notebooks with:

```powershell
jupyter lab notebooks
```

Suggested order:

```text
01_yfinance_data_pull.ipynb
02_baseline_model.ipynb
03_stock_momentum_strategy.ipynb
04_extended_universe_data_pull.ipynb
05_precious_metals_rotation.ipynb
06_volatility_risk_controls.ipynb
07_walk_forward_ml.ipynb
```
