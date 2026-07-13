from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.backtest import TRADING_DAYS


def choose_cost_aware_positions(
    expected_returns: pd.Series,
    *,
    mode: str,
    allowed_positions: tuple[float, ...] | None = None,
    full_position_edge: float | None = None,
    transaction_cost_bps: float = 1.0,
    short_borrow_cost_bps: float = 0.0,
    trading_days: int = TRADING_DAYS,
    initial_position: float = 0.0,
) -> pd.Series:
    """Choose daily positions by maximizing expected one-day net utility.

    The chooser compares each allowed position against the previous day's
    position, so switching from +1 to -1 pays two units of turnover.

    When ``full_position_edge`` is provided, positions are sized with a
    quadratic risk penalty. This makes fractional positions useful: a weak
    expected edge may choose +/-0.25 or cash instead of jumping to +/-1.
    """
    if allowed_positions is not None:
        candidates = tuple(sorted(float(position) for position in allowed_positions))
    elif mode == "long_cash":
        candidates = (0.0, 1.0)
    elif mode == "long_short":
        candidates = (-1.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported position mode: {mode}")

    if not candidates:
        raise ValueError("At least one allowed position is required.")
    if min(candidates) < -1.0 or max(candidates) > 1.0:
        raise ValueError("Allowed positions must stay inside [-1.0, 1.0].")
    if mode == "long_cash" and min(candidates) < 0.0:
        raise ValueError("long_cash mode cannot include short positions.")
    if full_position_edge is not None and full_position_edge <= 0:
        raise ValueError("full_position_edge must be positive when provided.")

    cost_rate = transaction_cost_bps / 10_000
    short_borrow_daily = short_borrow_cost_bps / 10_000 / trading_days

    positions: list[float] = []
    previous_position = float(initial_position if initial_position in candidates else 0.0)

    for expected_return in expected_returns.fillna(0.0):
        best_position = previous_position if previous_position in candidates else 0.0
        best_value = -np.inf

        for candidate in candidates:
            turnover = abs(candidate - previous_position)
            transaction_cost = turnover * cost_rate
            short_borrow_cost = abs(min(candidate, 0.0)) * short_borrow_daily
            position_penalty = 0.0
            if full_position_edge is not None:
                position_penalty = 0.5 * full_position_edge * candidate**2
            expected_net_return = (
                candidate * expected_return
                - position_penalty
                - transaction_cost
                - short_borrow_cost
            )

            better_value = expected_net_return > best_value + 1e-12
            similar_value = abs(expected_net_return - best_value) <= 1e-12
            lower_turnover = abs(candidate - previous_position) < abs(best_position - previous_position)

            if better_value or (similar_value and lower_turnover):
                best_position = candidate
                best_value = expected_net_return

        positions.append(best_position)
        previous_position = best_position

    return pd.Series(positions, index=expected_returns.index, name="position")
