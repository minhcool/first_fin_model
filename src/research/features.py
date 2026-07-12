from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_features(
    close: pd.DataFrame,
    *,
    target_asset: str = "SPY",
    return_windows: tuple[int, ...] = (5, 20, 60),
    volatility_windows: tuple[int, ...] = (5, 20, 60),
    correlation_windows: tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    close = close.sort_index()
    returns = close.pct_change()
    features = pd.DataFrame(index=close.index)

    for ticker in close.columns:
        features[f"{ticker}_ret_1d"] = returns[ticker]
        for window in return_windows:
            features[f"{ticker}_ret_{window}d"] = close[ticker].pct_change(window)
        for window in volatility_windows:
            features[f"{ticker}_vol_{window}d"] = returns[ticker].rolling(window).std() * np.sqrt(252)

    if target_asset in close.columns:
        for ticker in [ticker for ticker in close.columns if ticker != target_asset]:
            for window in correlation_windows:
                features[f"{target_asset}_{ticker}_corr_{window}d"] = (
                    returns[target_asset].rolling(window).corr(returns[ticker])
                )

        rolling_high = close[target_asset].rolling(252, min_periods=60).max()
        features[f"{target_asset}_drawdown_252d"] = close[target_asset] / rolling_high - 1
        ma_200 = close[target_asset].rolling(200).mean()
        features[f"{target_asset}_above_200d_avg"] = (
            close[target_asset] > ma_200
        ).astype(float).where(ma_200.notna())

    for ticker in close.columns:
        rolling_high = close[ticker].rolling(252, min_periods=60).max()
        features[f"{ticker}_drawdown_252d"] = close[ticker] / rolling_high - 1

    return features


def make_next_day_direction_dataset(close: pd.DataFrame, *, target_asset: str = "SPY") -> pd.DataFrame:
    returns = close.pct_change()
    features = build_market_features(close, target_asset=target_asset)
    target_return = returns[target_asset].shift(-1).rename(f"{target_asset}_next_day_return")
    target_up = (target_return > 0).astype(float).rename(f"{target_asset}_next_day_up")
    dataset = pd.concat([features, target_return, target_up], axis=1)
    return dataset.dropna()

