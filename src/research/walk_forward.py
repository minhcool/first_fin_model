from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_index: pd.Index
    test_index: pd.Index


def make_walk_forward_splits(
    dates: pd.Series,
    *,
    train_months: int = 36,
    test_months: int = 6,
    step_months: int = 6,
) -> list[WalkForwardSplit]:
    dates = pd.Series(pd.to_datetime(dates)).sort_values()
    first_date = dates.min()
    last_date = dates.max()

    splits: list[WalkForwardSplit] = []
    train_start = first_date

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_start > last_date:
            break

        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= test_start) & (dates < test_end)

        if train_mask.sum() > 50 and test_mask.sum() > 10:
            splits.append(
                WalkForwardSplit(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=min(test_end, last_date),
                    train_index=dates[train_mask].index,
                    test_index=dates[test_mask].index,
                )
            )

        train_start = train_start + pd.DateOffset(months=step_months)
        if train_start + pd.DateOffset(months=train_months) > last_date:
            break

    return splits

