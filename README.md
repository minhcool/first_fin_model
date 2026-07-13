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

The original ML trading setup is a daily long/cash SPY timing strategy:

- If the model predicts SPY up tomorrow: hold `100% SPY`
- If the model predicts SPY down tomorrow: hold `100% cash`
- No shorting
- No leverage
- No intraday trading
- Signals are evaluated out of sample with walk-forward splits

## Long/Short Research Direction

The next research step is to test a long/short version of the model:

- If the model is bullish: hold `+100% SPY`
- If the model is bearish: hold `-100% SPY`
- If a neutral threshold is added later: hold `0% cash`

This is a research extension, not a live-trading-ready assumption. A short position is more complicated than a long position because the backtest must eventually model:

- Whether shares are available to borrow
- Short borrow fees or hard-to-borrow costs
- Margin requirements and possible margin calls
- Financing cost on borrowed exposure
- Realistic execution price, such as next open, close auction, or VWAP
- The fact that a switch from `+100%` long to `-100%` short is a `200%` notional turnover event

For now, the first long/short implementation treats `+1.0` as fully long, `0.0` as cash, and `-1.0` as fully short. This lets the research pipeline measure whether bearish predictions are useful before adding a more realistic broker/margin model.

## Assumptions

- Transaction cost: `1.0` basis point per amount traded in the ML notebook
- `1` basis point is `1 / 10,000`, or `0.01%`, of traded notional
- Daily transaction cost is `turnover * transaction_cost_bps / 10,000`
- Total transaction cost is approximately `annual_turnover * years * transaction_cost_bps / 10,000`
- A full switch from cash to SPY costs `0.01%`
- A full round trip from cash to SPY and back to cash costs about `0.02%`
- Preliminary short borrow cost for long/short research: `25` basis points per year when short
- No taxes are modeled
- No bid-ask spread or slippage is modeled beyond the transaction-cost assumption
- Current execution assumption: signal uses information through today's daily close and applies to the next close-to-close return
- The current walk-forward ML notebook subtracts transaction costs in the realized backtest and uses a cost-aware decision rule before trading
- Results use historical daily data and should not be treated as live trading performance

The cost-aware decision rule does not make the classifier itself a fully cost-optimized trading model. The classifier still learns next-day direction. After training, the decision layer estimates next-day SPY return from the model's score and the training split's average up/down return sizes, then chooses long, cash, or short after subtracting estimated transaction and short-borrow costs. A more advanced future version could train directly on expected net return instead of direction.

## Current Models

The walk-forward ML notebook tests:

- Logistic Regression
- Ridge Classifier
- Random Forest
- Gradient Boosting

The current best-performing model in `notebooks/07_walk_forward_ml.ipynb` is `logistic_regression`, but this should be treated as a research result, not proof that the model is robust.

## Fair Comparison Note

The older rule-strategy notebooks originally reported results over all available daily history, roughly from `2015` to `2026`. The ML notebook is stricter: it uses walk-forward testing where each model trains on the prior `36` months, then trades only the next `6` months out of sample.

That means the original rule-strategy results and the ML results were not fully apples-to-apples. The rule strategies were judged across older market regimes too, while the ML results were judged only on rolling out-of-sample test windows starting on `2018-10-16`.

To make the comparison fairer, the older daily rule-strategy outputs were re-scored on the exact same ML test calendar:

- `16` test windows
- `1,937` ML out-of-sample test days
- First test date: `2018-10-16`
- Last test date: `2026-07-02`
- Aggregate report: `reports/walk_forward_strategy_comparison.csv`
- Split-by-split report: `reports/walk_forward_strategy_split_results.csv`
- Simple spreadsheet report: `reports/walk_forward_strategy_comparison_simple.xlsx`

The rule strategies are not retrained because they are rule-based rather than fitted models. For them, the fairer comparison is to filter their generated daily returns to the same out-of-sample test dates used by the ML model. Some older stock and baseline files have slightly fewer rows than `1,937` because their source feature history does not cover every ML test date.

## Smaller Walk-Forward Windows

Shorter walk-forward windows can work, and they are a useful next robustness test. The current `36` month train / `6` month test setup is stable enough to estimate risk metrics, but it may be slow to notice when an alpha stops working.

Useful future tests:

- `36` month train / `3` month test
- `24` month train / `1` month test
- `12` month train / `1` month test
- Possibly `6` month train / `2` week test for very short-lived signals

The tradeoff is noise. With daily data, a week has only about `5` trading rows and a month has only about `21` trading rows, so Sharpe, drawdown, and win rate become much less reliable on a single split. Short windows should be judged across many repeated splits, not from one good or bad period.

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
- Short selling is only a preliminary research mode and does not yet fully model broker-specific margin, borrow availability, or hard-to-borrow fees
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
