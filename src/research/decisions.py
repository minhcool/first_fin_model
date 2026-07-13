from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.backtest import TRADING_DAYS


def choose_cost_aware_positions(
    expected_returns: pd.Series,
    *,
    mode: str,
    transaction_cost_bps: float = 1.0,
    short_borrow_cost_bps: float = 0.0,
    trading_days: int = TRADING_DAYS,
    initial_position: float = 0.0,
) -> pd.Series:
    """Choose daily positions by maximizing expected one-day net return.

    The chooser compares each allowed position against the previous day's
    position, so switching from +1 to -1 pays two units of turnover.
    """
    if mode == "long_cash":
        candidates = (0.0, 1.0)
    elif mode == "long_short":
        candidates = (-1.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported position mode: {mode}")

    cost_rate = transaction_cost_bps / 10_000
    short_borrow_daily = short_borrow_cost_bps / 10_000 / trading_days

    positions: list[float] = []
    previous_position = float(initial_position)

    for expected_return in expected_returns.fillna(0.0):
        best_position = previous_position if previous_position in candidates else 0.0
        best_value = -np.inf

        for candidate in candidates:
            turnover = abs(candidate - previous_position)
            transaction_cost = turnover * cost_rate
            short_borrow_cost = abs(min(candidate, 0.0)) * short_borrow_daily
            expected_net_return = candidate * expected_return - transaction_cost - short_borrow_cost

            better_value = expected_net_return > best_value + 1e-12
            similar_value = abs(expected_net_return - best_value) <= 1e-12
            lower_turnover = abs(candidate - previous_position) < abs(best_position - previous_position)

            if better_value or (similar_value and lower_turnover):
                best_position = candidate
                best_value = expected_net_return

        positions.append(best_position)
        previous_position = best_position

    return pd.Series(positions, index=expected_returns.index, name="position")
