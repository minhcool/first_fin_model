from __future__ import annotations

import numpy as np
import pandas as pd


def empty_weights(close: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=close.index, columns=close.columns)


def buy_and_hold_weights(close: pd.DataFrame, asset: str) -> pd.DataFrame:
    weights = empty_weights(close)
    weights[asset] = 1.0
    return weights


def long_cash_momentum_weights(close: pd.DataFrame, asset: str, window: int) -> pd.DataFrame:
    weights = empty_weights(close)
    momentum = close[asset].pct_change(window)
    weights.loc[momentum > 0, asset] = 1.0
    return weights


def equal_positive_momentum_weights(close: pd.DataFrame, assets: list[str], window: int) -> pd.DataFrame:
    weights = empty_weights(close)
    momentum = close[assets].pct_change(window)
    positive = momentum > 0
    positive_count = positive.sum(axis=1).replace(0, np.nan)

    for asset in assets:
        weights.loc[positive[asset], asset] = 1 / positive_count[positive[asset]]

    return weights.fillna(0.0)


def top_n_momentum_weights(
    close: pd.DataFrame,
    assets: list[str],
    window: int,
    *,
    top_n: int = 1,
    require_positive: bool = True,
) -> pd.DataFrame:
    weights = empty_weights(close)
    momentum = close[assets].pct_change(window)

    for date, row in momentum.iterrows():
        row = row.dropna().sort_values(ascending=False)
        if require_positive:
            row = row[row > 0]
        selected = row.head(top_n).index.tolist()
        if selected:
            weights.loc[date, selected] = 1 / len(selected)

    return weights


def volatility_target_weights(
    close: pd.DataFrame,
    base_weights: pd.DataFrame,
    *,
    target_volatility: float = 0.12,
    volatility_window: int = 20,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    returns = close.pct_change()
    portfolio_returns = (base_weights.reindex(close.index).fillna(0.0) * returns).sum(axis=1)
    realized_volatility = portfolio_returns.rolling(volatility_window).std() * np.sqrt(252)
    scale = (target_volatility / realized_volatility).clip(upper=max_leverage)
    return base_weights.mul(scale, axis=0).fillna(0.0)


def apply_trailing_stop(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    asset: str,
    volatility_window: int = 20,
    volatility_multiplier: float = 3.0,
) -> pd.DataFrame:
    adjusted = weights.copy().fillna(0.0)
    returns = close[asset].pct_change()
    daily_volatility = returns.rolling(volatility_window).std()

    in_position = False
    high_since_entry = np.nan

    for date in close.index:
        wants_position = adjusted.at[date, asset] > 0
        price = close.at[date, asset]
        vol = daily_volatility.at[date]

        if not wants_position:
            in_position = False
            high_since_entry = np.nan
            continue

        if not in_position:
            in_position = True
            high_since_entry = price
            continue

        high_since_entry = max(high_since_entry, price)
        stop_distance = volatility_multiplier * vol
        trailing_drawdown = price / high_since_entry - 1

        if pd.notna(stop_distance) and trailing_drawdown <= -stop_distance:
            adjusted.at[date, asset] = 0.0
            in_position = False
            high_since_entry = np.nan

    return adjusted

