# Predictions

Starter workspace for financial market prediction experiments.

## Project Layout

- `notebooks/` - exploratory notebooks and modeling experiments
- `src/data/` - reusable data download/cleaning code
- `data/raw/` - downloaded source data from `yfinance`
- `data/processed/` - cleaned datasets ready for modeling

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Download Starter Market Data

```powershell
python -m src.data.download_yfinance --tickers SPY QQQ TLT GLD --start 2015-01-01
```

That writes a CSV into `data/raw/`.

## Notebooks

Open the starter notebook with:

```powershell
jupyter lab notebooks
```

Start with `notebooks/01_yfinance_data_pull.ipynb`.
