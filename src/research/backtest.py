from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    equity_curve = (1 + returns.fillna(0.0)).cumprod()
    running_high = equity_curve.cummax()
    drawdown = equity_curve / running_high - 1
    return float(drawdown.min())


def performance_metrics(
    returns: pd.Series,
    *,
    name: str,
    exposure: pd.Series | None = None,
    turnover: pd.Series | None = None,
    transaction_cost: pd.Series | None = None,
    trading_days: int = TRADING_DAYS,
) -> dict[str, float | str]:
    returns = returns.dropna()
    years = len(returns) / trading_days
    total_return = float((1 + returns).prod() - 1)
    annual_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else np.nan
    annual_volatility = float(returns.std(ddof=0) * np.sqrt(trading_days))
    sharpe = np.nan if annual_volatility == 0 else float(returns.mean() / returns.std(ddof=0) * np.sqrt(trading_days))
    drawdown = max_drawdown(returns)
    calmar = np.nan if drawdown == 0 else float(annual_return / abs(drawdown))

    metrics: dict[str, float | str] = {
        "strategy": name,
        "rows": len(returns),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "positive_day_rate": float((returns > 0).mean()),
    }

    if exposure is not None:
        metrics["avg_exposure"] = float(exposure.reindex(returns.index).mean())
    if turnover is not None:
        metrics["annual_turnover"] = float(turnover.reindex(returns.index).mean() * trading_days)
    if transaction_cost is not None:
        metrics["total_transaction_cost"] = float(transaction_cost.reindex(returns.index).sum())

    return metrics


def backtest_weights(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    name: str,
    transaction_cost_bps: float = 1.0,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    close = close.sort_index()
    returns = close.pct_change()
    forward_returns = returns.shift(-1)

    weights = weights.reindex(close.index).fillna(0.0)
    weights = weights.reindex(columns=close.columns, fill_value=0.0)

    gross_returns = (weights * forward_returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * (transaction_cost_bps / 10_000)
    net_returns = gross_returns - costs

    daily = pd.DataFrame(
        {
            "Date": close.index,
            "strategy": name,
            "gross_return": gross_returns,
            "transaction_cost": costs,
            "strategy_return": net_returns,
            "turnover": turnover,
            "exposure": weights.abs().sum(axis=1),
        }
    ).dropna(subset=["strategy_return"])
    daily = daily.set_index("Date", drop=False)
    daily["equity_curve"] = (1 + daily["strategy_return"]).cumprod()

    metrics = performance_metrics(
        daily["strategy_return"],
        name=name,
        exposure=daily["exposure"],
        turnover=daily["turnover"],
        transaction_cost=daily["transaction_cost"],
        trading_days=trading_days,
    )
    return daily.reset_index(drop=True), metrics


def backtest_positions(
    actual_returns: pd.Series,
    positions: pd.Series,
    *,
    name: str,
    transaction_cost_bps: float = 1.0,
    short_borrow_cost_bps: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    actual_returns = actual_returns.sort_index()
    positions = positions.reindex(actual_returns.index).fillna(0.0).astype(float)

    turnover = positions.diff().abs().fillna(positions.abs())
    transaction_cost = turnover * (transaction_cost_bps / 10_000)
    short_exposure = positions.clip(upper=0).abs()
    short_borrow_cost = short_exposure * (short_borrow_cost_bps / 10_000 / trading_days)
    strategy_returns = positions * actual_returns - transaction_cost - short_borrow_cost

    daily = pd.DataFrame(
        {
            "Date": actual_returns.index,
            "strategy": name,
            "actual_return": actual_returns,
            "position": positions,
            "gross_return": positions * actual_returns,
            "transaction_cost": transaction_cost,
            "short_borrow_cost": short_borrow_cost,
            "strategy_return": strategy_returns,
            "turnover": turnover,
            "exposure": positions.abs(),
            "net_exposure": positions,
            "short_exposure": short_exposure,
        }
    ).dropna(subset=["strategy_return"])
    daily = daily.set_index("Date", drop=False)
    daily["equity_curve"] = (1 + daily["strategy_return"]).cumprod()

    metrics = performance_metrics(
        daily["strategy_return"],
        name=name,
        exposure=daily["exposure"],
        turnover=daily["turnover"],
        transaction_cost=daily["transaction_cost"],
        trading_days=trading_days,
    )
    metrics["avg_net_exposure"] = float(daily["net_exposure"].mean())
    metrics["short_day_rate"] = float((daily["position"] < 0).mean())
    metrics["long_day_rate"] = float((daily["position"] > 0).mean())
    metrics["cash_day_rate"] = float((daily["position"] == 0).mean())
    metrics["total_short_borrow_cost"] = float(daily["short_borrow_cost"].sum())
    return daily.reset_index(drop=True), metrics


def format_percent_table(frame: pd.DataFrame, percent_cols: list[str]) -> pd.DataFrame:
    display_frame = frame.copy()
    for col in percent_cols:
        if col in display_frame:
            display_frame[col] = display_frame[col].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
    return display_frame
